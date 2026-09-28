#!/usr/bin/env python3
"""Experiment SM-D: closed-loop state-gate cache runner.

docs/research_plan_stateful_marginal.md §5 SM-D. Uses the SM-C-fitted
estimator (est2 = q_n + log1p(C-state) features) — or q_n raw, or a fixed
interval baseline — to gate caching online, generate images, and log
per-step decisions. The forward computes the SAME online features as
SM-B's marginal_risk_probe (SEA-filtered modulated-input drift, q_n via
lib/marginal_features, plus C^stale/traj/mem), so the SM-C-fitted
estimator is in-distribution.

Gate rule:

  cache  iff  predict_n < threshold  AND  gap_n < g_max

where predict is:

  --gate q       : predict = q_n
                   q_n = P_hat * S_hat_n * |h_n| * A_hat_n
                   --p_mode path uses accumulated adjacent psi drift
                   --p_mode displacement uses rel_l1(psi_n, psi_a), where
                   a is the last full-refresh step.
                   --p_mode mix uses a geometric interpolation between the
                   path and displacement P factors:
                     P_mix = exp(beta log(P_path+eps)
                                 + (1-beta) log(P_disp+eps)).
                   --q_variant optionally ablates factors in the q score
                   while leaving logged q_n as the full score.
  --gate est2    : predict = exp(beta @ design)
                   design = [1, log(q+ε), g, n, n·g,
                            log1p(c_stale), log1p(c_traj), log1p(c_mem)]
                   beta loaded from marginal_estimator_summary.json
                   (analysis/marginal_estimator.py output of SM-C).
  --gate interval: cnt % interval != 0; ignores threshold (baseline).

The default cache payload is zero-order whole-transformer residual reuse
(matches SeaCache / TeaCache). For research-only payload ablations, cached
steps can instead use a TaylorSeer/HiCache-style forecast computed only from
past full-refresh residual history. Native SeaCache / TeaCache baselines remain
in flux/seacache.py and flux/teacache.py; this runner adds q/est2 gates and
research payload variants outside the locked baselines.

Output per prompt:
  prompt_XXXXX/image.png        PIL image (the generated sample)
  prompt_XXXXX/decisions.json   per-step {step, u, predict, q, gap,
                                          c_stale, c_traj, c_mem}
  prompt_XXXXX/manifest.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from lib.history_fd_observer import (  # noqa: E402
    clone_state as clone_history_fd_state,
    ffro_residual_fields as history_fd_ffro_residual_fields,
    forecast_predictions as history_fd_forecast_predictions,
    init_state as init_history_fd_state,
    online_fields as history_fd_online_fields,
    update_on_full as history_fd_update_on_full,
)
from lib.sencache import load_sensitivity_table, online_fields as sencache_online_fields  # noqa: E402
from lib.teacache_coeffs import get_coeffs  # noqa: E402
from lib.wiener import apply_sea_with_scheduler  # noqa: E402
from lib.marginal_features import MarginalFeatureTracker  # noqa: E402

logger = logging.get_logger(__name__)

EPS = 1e-6


def _norm_optional(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.detach().to(torch.float32).norm().item())


def _norm(t: torch.Tensor) -> float:
    return float(t.detach().to(torch.float32).norm().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    an = _norm(a)
    bn = _norm(b)
    if an <= 0.0 or bn <= 0.0:
        return None
    dot = float(torch.sum(a.detach().to(torch.float32) * b.detach().to(torch.float32)).item())
    return dot / (an * bn + EPS)


# ---------------------------------------------------------------------------
# Estimator loading + prediction (matches analysis/marginal_estimator.py _design)
# ---------------------------------------------------------------------------
def _load_est_coef(path: Path, est: str) -> List[float]:
    summary = json.loads(Path(path).read_text())
    coef = summary["estimators"][est]["coef"]
    if coef is None:
        raise SystemExit(f"estimator {est!r} has no coefficients in {path}")
    return [float(x) for x in coef]


def _load_proxy_model(
    path: Optional[Path],
    *,
    expected_family: Optional[str] = None,
    allowed_families: Optional[Tuple[str, ...]] = None,
) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "online_proxy_gate_model.v1":
        raise SystemExit(f"unsupported proxy model schema in {path}: {data.get('schema')!r}")
    family = str(data.get("family"))
    if expected_family is not None and family != expected_family:
        raise SystemExit(
            f"proxy model family {family!r} does not match gate {expected_family!r}"
        )
    if allowed_families is not None and family not in allowed_families:
        raise SystemExit(
            f"proxy model family {family!r} is not one of {list(allowed_families)!r}"
        )
    if family != "shadow" and data.get("online_status") is not True:
        raise SystemExit(f"{family} proxy model is not strict zero-extra online")
    if family == "shadow" and data.get("online_status") != "partial-forward":
        raise SystemExit("shadow proxy model must be marked partial-forward")
    model_class = str(data.get("model_class") or "ridge")
    if model_class not in {"ridge", "mlp", "gbdt"}:
        raise SystemExit(f"unsupported proxy model_class in {path}: {model_class!r}")
    if model_class == "ridge":
        coef = data.get("coef")
        if not isinstance(coef, list) or not coef:
            raise SystemExit(f"ridge proxy model has no coefficient vector: {path}")
    else:
        head = data.get("head")
        if not isinstance(head, dict):
            raise SystemExit(f"{model_class} proxy model has no serialized head: {path}")
    return data


def _proxy_value(row: Dict[str, Any], col: str) -> Optional[float]:
    value = row.get(col)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _proxy_predict_log1p(model: Dict[str, Any], row: Dict[str, Any]) -> float:
    design = model["design"]
    missing = [
        col
        for col in list(design["cat_cols"]) + list(design["cont_cols"])
        if col not in row
    ]
    if missing:
        family = model.get("family")
        raise RuntimeError(f"proxy model family {family!r} missing design columns: {missing[:20]}")
    vals: List[float] = [1.0]
    for col in design["cat_cols"]:
        current = str(row.get(col, ""))
        for level in design["cat_levels"][col]:
            vals.append(1.0 if current == str(level) else 0.0)
    for col in design["cont_cols"]:
        stats = design["cont_stats"][col]
        x = _proxy_value(row, col)
        if x is None:
            x = float(stats["fill"])
        vals.append((float(x) - float(stats["mean"])) / (float(stats["std"]) or 1.0))
    model_class = str(model.get("model_class") or "ridge")
    if model_class == "ridge":
        coef = [float(v) for v in model["coef"]]
        if len(vals) != len(coef):
            raise RuntimeError(f"proxy design length {len(vals)} != coef length {len(coef)}")
        return float(sum(v * b for v, b in zip(vals, coef)))
    if model_class == "mlp":
        x = np.asarray(vals, dtype=float).reshape(1, -1)
        for layer in model["head"]["layers"]:
            w = np.asarray(layer["weight"], dtype=float)
            b = np.asarray(layer["bias"], dtype=float)
            x = x @ w.T + b
            if layer.get("activation") == "relu":
                x = np.maximum(x, 0.0)
        return float(x.reshape(-1)[0])
    if model_class == "gbdt":
        x = vals
        pred = float(model["head"]["init"])
        lr = float(model["head"]["learning_rate"])
        for tree in model["head"]["trees"]:
            nodes = tree["nodes"]
            node_idx = 0
            while True:
                node = nodes[node_idx]
                if node.get("leaf"):
                    pred += lr * float(node["value"])
                    break
                feature = int(node["feature"])
                threshold = float(node["threshold"])
                if node.get("operator", "lt") == "lt":
                    go_left = float(x[feature]) < threshold
                else:
                    go_left = float(x[feature]) <= threshold
                node_idx = int(node["left"] if go_left else node["right"])
        return float(pred)
    raise RuntimeError(f"unsupported proxy model_class: {model_class!r}")


def _parse_int_set(text: str) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for token in text.replace(",", " ").split():
        out.add(int(token))
    return out


def _validate_proxy_step_support(
    model: Dict[str, Any],
    *,
    action_steps: set[int],
    num_steps: int,
    allow_sparse_numeric_proxy: bool = False,
) -> None:
    if model.get("step_encoding") != "categorical":
        if (
            not action_steps
            and model.get("step_encoding") == "numeric"
            and not bool(model.get("dense_all_step_labels"))
            and not allow_sparse_numeric_proxy
        ):
            raise SystemExit(
                f"numeric proxy model family {model.get('family')!r} is being deployed "
                "all-step but does not record dense_all_step_labels=true. Fit from "
                "dense all-step rows or pass --allow_sparse_numeric_proxy for an "
                "explicit extrapolation-only diagnostic."
            )
        return
    seen = {
        int(v)
        for v in model.get("fork_step_values", [])
    }
    if action_steps:
        missing = sorted(int(s) for s in action_steps if int(s) not in seen)
        if missing:
            raise SystemExit(
                f"categorical proxy model family {model.get('family')!r} was not trained "
                f"on requested action_steps: {missing[:20]}"
            )
        return
    eligible = set(range(1, max(int(num_steps) - 1, 1)))
    missing = sorted(eligible - seen)
    if missing:
        raise SystemExit(
            f"categorical proxy model family {model.get('family')!r} covers only "
            f"{sorted(seen)}. Pass --action_steps with trained steps, or fit with "
            f"--step_encoding numeric for all-step deployment."
        )


def _validate_proxy_context(model: Dict[str, Any], *, proxy_mode: str, proxy_cache_threshold: float) -> None:
    mode_values = {str(v) for v in model.get("mode_values", [])}
    if mode_values and str(proxy_mode) not in mode_values:
        raise SystemExit(
            f"proxy model family {model.get('family')!r} was not trained on "
            f"proxy_mode={proxy_mode!r}; trained modes={sorted(mode_values)!r}"
        )
    threshold_values = model.get("cache_threshold_values", [])
    if threshold_values:
        ok = False
        for value in threshold_values:
            try:
                if abs(float(value) - float(proxy_cache_threshold)) <= 1e-9:
                    ok = True
                    break
            except (TypeError, ValueError):
                if str(value) == str(proxy_cache_threshold):
                    ok = True
                    break
        if not ok:
            raise SystemExit(
                f"proxy model family {model.get('family')!r} was not trained on "
                f"proxy_cache_threshold={proxy_cache_threshold}; trained thresholds={threshold_values!r}"
            )


Q_VARIANTS = ("full", "p", "ph", "ps", "pa", "psh", "pha", "psa",
              "v_only", "shuffled_p", "senqa")

P_MODES = ("path", "displacement", "mix")

PAYLOAD_MODES = (
    "reuse",
    "taylor_o1",
    "taylor_o2",
    "hicache_o2",
    "ensemble_mean",
)

FORMULA_VARIANTS = (
    "q_full",
    "q_native",
    "ffro_res",
    "ffro_step",
    "ffro_gv",
    "q_ffro_res_veto",
    "q_ffro_gv_veto",
)

FFRO_FORMULA_VARIANTS = {
    "ffro_res",
    "ffro_step",
    "ffro_gv",
    "q_ffro_res_veto",
    "q_ffro_gv_veto",
}
PERCENTILE_FORMULA_VARIANTS = {
    "q_ffro_res_veto",
    "q_ffro_gv_veto",
}


def _est2_predict(q: float, gap: int, step: int, c_stale: float,
                  c_traj: float, c_mem: float, beta: List[float]) -> float:
    """Δ̂ = exp(beta @ design). Layout must match marginal_estimator._design."""
    design = [1.0, math.log(q + EPS), float(gap), float(step),
              float(step) * float(gap),
              math.log1p(c_stale), math.log1p(c_traj), math.log1p(c_mem)]
    if len(beta) != len(design):
        raise SystemExit(f"est2 beta length {len(beta)} != design {len(design)}")
    return math.exp(sum(b * d for b, d in zip(beta, design)))


def _q_variant_score(variant: str, *, p: float, s: float, h: float, a: float,
                     sen: Optional[float] = None,
                     p_shuf: Optional[float] = None) -> float:
    """Return the q-gate decision score for a factor-ablation variant.

    `full` is the historical q-gate score P*S*|h|*A. Other variants are
    deployment controls; `feats["q"]` remains the full score for logging and
    est2 compatibility.
    """
    if variant == "full":
        return p * s * h * a
    if variant == "p":
        return p
    if variant == "ph":
        return p * h
    if variant == "ps":
        return p * s
    if variant == "pa":
        return p * a
    if variant == "psh":
        return p * s * h
    if variant == "pha":
        return p * h * a
    if variant == "psa":
        return p * s * a
    if variant == "v_only":
        return s * h * a
    if variant == "shuffled_p":
        if p_shuf is None:
            raise ValueError("q_variant='shuffled_p' requires p_shuf")
        return p_shuf * s * h * a
    if variant == "senqa":
        if sen is None:
            raise ValueError("q_variant='senqa' requires SenCache score")
        return sen * h * a
    raise ValueError(f"unknown q_variant: {variant!r}")


def _q_variant_uses(variant: str) -> Dict[str, bool]:
    """Boolean factor-use flags for decisions.json auditing."""
    return {
        "use_p": variant not in ("v_only", "senqa"),
        "use_s": variant in ("full", "ps", "psh", "psa", "v_only", "shuffled_p"),
        "use_h": variant in ("full", "ph", "psh", "pha", "v_only", "shuffled_p", "senqa"),
        "use_a": variant in ("full", "pa", "pha", "psa", "v_only", "shuffled_p", "senqa"),
        "use_sencache": variant in ("senqa",),
    }


def _p_mix(path_value: float, disp_value: float, beta: float) -> float:
    beta = float(beta)
    return math.exp(
        beta * math.log(max(float(path_value), 0.0) + EPS)
        + (1.0 - beta) * math.log(max(float(disp_value), 0.0) + EPS)
    )


def _forecast_payload(
    mode: str,
    *,
    reuse: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    step: int,
    sigma: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return the residual payload used on a cached step.

    Forecast modes are decision-time legal: they depend only on residuals from
    prior full-refresh steps recorded in `history_state`. If the requested
    forecast is unavailable, the function falls back to reuse and records that
    fallback explicitly.
    """
    if mode == "reuse":
        return reuse, {
            "payload_mode": "reuse",
            "payload_used": "reuse",
            "payload_available": True,
            "payload_fallback": False,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": None,
            "payload_delta_from_reuse_norm": 0.0,
        }
    preds = history_fd_forecast_predictions(history_state, step=int(step), sigma=float(sigma))
    forecast: Optional[torch.Tensor]
    if mode == "ensemble_mean":
        parts = [
            preds.get(name)
            for name in ("taylor_o1", "taylor_o2", "hicache_o2")
            if preds.get(name) is not None and preds.get(name).shape == reuse.shape
        ]
        forecast = None if not parts else torch.stack([p.to(reuse.dtype) for p in parts], dim=0).mean(dim=0)
    else:
        forecast = preds.get(mode)
    if forecast is None or forecast.shape != reuse.shape:
        return reuse, {
            "payload_mode": str(mode),
            "payload_used": "reuse",
            "payload_available": False,
            "payload_fallback": True,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": None,
            "payload_delta_from_reuse_norm": 0.0,
        }
    forecast = forecast.to(dtype=reuse.dtype, device=reuse.device)
    return forecast, {
        "payload_mode": str(mode),
        "payload_used": str(mode),
        "payload_available": True,
        "payload_fallback": False,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": _norm(forecast),
        "payload_delta_from_reuse_norm": _norm(forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)),
    }


def _load_formula_table(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "ffro_formula_percentile_table.v1":
        raise SystemExit(f"unsupported formula table schema: {data.get('schema')!r}")
    return data


def _formula_group_keys(mode: str, threshold: float, step: int) -> List[str]:
    mode_s = str(mode)
    thresh_s = f"{float(threshold):.12g}"
    step_i = int(step)
    return [
        f"{mode_s}|{thresh_s}|{step_i}",
        f"*|*|{step_i}",
        "*|*|*",
    ]


def _percentile_from_table(
    table: Optional[Dict[str, Any]],
    *,
    score_name: str,
    score_value: Optional[float],
    mode: str,
    threshold: float,
    step: int,
) -> Optional[float]:
    if table is None or score_value is None or not math.isfinite(float(score_value)):
        return None
    groups = table.get("groups") or {}
    for key in _formula_group_keys(mode, threshold, step):
        vals = groups.get(key, {}).get(score_name)
        if not vals:
            continue
        arr = [float(v) for v in vals if math.isfinite(float(v))]
        if not arr:
            continue
        pos = bisect.bisect_right(arr, float(score_value))
        return float(pos / len(arr))
    return None


def _first_finite(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        val = mapping.get(key)
        if val is None:
            continue
        try:
            x = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            return x
    return None


def _formula_scores(
    variant: str,
    *,
    proxy_row: Dict[str, Any],
    score_full: float,
    s_factor: float,
    h_abs: float,
    a_factor: float,
    formula_table: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute pre-registered FFRO/q formula scores from decision-time fields."""
    q_native_no_sa = _first_finite(proxy_row, ("online_q_proxy_no_sa",))
    q_native = None if q_native_no_sa is None else float(q_native_no_sa) * float(s_factor) * float(a_factor)

    ffro_reuse = _first_finite(proxy_row, ("online_ffro_reuse_residual_norm_pre",))
    ffro_res_abs = _first_finite(
        proxy_row,
        (
            "online_ffro_ensemble_res_max_pre",
            "online_ffro_taylor_o2_res_norm_pre",
            "online_ffro_hicache_o2_res_norm_pre",
            "online_ffro_taylor_o1_res_norm_pre",
        ),
    )
    ffro_res_rel_existing = _first_finite(
        proxy_row,
        (
            "online_ffro_taylor_o2_res_rel_pre",
            "online_ffro_hicache_o2_res_rel_pre",
            "online_ffro_taylor_o1_res_rel_pre",
        ),
    )
    ffro_res_rel = ffro_res_rel_existing
    if ffro_res_rel is None and ffro_res_abs is not None and ffro_reuse is not None:
        ffro_res_rel = float(ffro_res_abs) / (float(ffro_reuse) + EPS)
    ffro_res = (
        None if ffro_res_rel is None
        else float(ffro_res_rel) * float(h_abs) * float(s_factor) * float(a_factor)
    )

    ffro_step = _first_finite(
        proxy_row,
        (
            "online_ffro_ensemble_step_max_pre",
            "online_ffro_taylor_o2_step_norm_pre",
            "online_ffro_hicache_o2_step_norm_pre",
            "online_ffro_taylor_o1_step_norm_pre",
        ),
    )
    ffro_gv = None if ffro_step is None else float(ffro_step) * float(a_factor)

    mode = str(proxy_row.get("mode", ""))
    threshold = float(proxy_row.get("cache_threshold", 0.0))
    step = int(proxy_row.get("step_index", proxy_row.get("fork_step", 0)))
    q_pct = _percentile_from_table(
        formula_table, score_name="q_native", score_value=q_native,
        mode=mode, threshold=threshold, step=step,
    )
    ffro_res_pct = _percentile_from_table(
        formula_table, score_name="ffro_res", score_value=ffro_res,
        mode=mode, threshold=threshold, step=step,
    )
    ffro_gv_pct = _percentile_from_table(
        formula_table, score_name="ffro_gv", score_value=ffro_gv,
        mode=mode, threshold=threshold, step=step,
    )

    predict: Optional[float]
    score_name: str
    cost_class = "zero_extra"
    if variant == "q_full":
        predict = float(score_full)
        score_name = "q_full"
    elif variant == "q_native":
        predict = q_native
        score_name = "q_native"
    elif variant == "ffro_res":
        predict = ffro_res
        score_name = "ffro_res"
    elif variant == "ffro_step":
        predict = ffro_step
        score_name = "ffro_step"
        cost_class = "velocity_head_extra_compute"
    elif variant == "ffro_gv":
        predict = ffro_gv
        score_name = "ffro_gv"
        cost_class = "velocity_head_extra_compute"
    elif variant == "q_ffro_res_veto":
        predict = None if q_pct is None or ffro_res_pct is None else max(float(q_pct), float(ffro_res_pct))
        score_name = "max_percentile(q_native,ffro_res)"
    elif variant == "q_ffro_gv_veto":
        predict = None if q_pct is None or ffro_gv_pct is None else max(float(q_pct), float(ffro_gv_pct))
        score_name = "max_percentile(q_native,ffro_gv)"
        cost_class = "velocity_head_extra_compute"
    else:
        raise ValueError(f"unknown formula_variant: {variant!r}")

    return {
        "formula_variant": str(variant),
        "formula_score_name": score_name,
        "formula_predict": predict,
        "formula_available": bool(predict is not None and math.isfinite(float(predict))),
        "formula_cost_class": cost_class,
        "formula_extra_compute": bool(cost_class != "zero_extra"),
        "formula_q_native": q_native,
        "formula_ffro_res": ffro_res,
        "formula_ffro_step": ffro_step,
        "formula_ffro_gv": ffro_gv,
        "formula_q_native_percentile": q_pct,
        "formula_ffro_res_percentile": ffro_res_pct,
        "formula_ffro_gv_percentile": ffro_gv_pct,
        "formula_table_schema": None if formula_table is None else formula_table.get("schema"),
    }


def _load_p_shuffle_table(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    if "values" not in data:
        raise SystemExit(f"p shuffle table missing 'values': {path}")
    return data


def _stable_offset(seed: int, step: int, n: int) -> int:
    if n <= 1:
        return 0
    # Simple deterministic non-cryptographic mix, stable across processes.
    x = (int(seed) + 0x9E3779B9) ^ ((int(step) + 1) * 0x85EBCA6B)
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    off = int(x % n)
    return off if off != 0 else 1


def _lookup_shuffled_p(table: Optional[Dict[str, Any]], *, prompt_idx: int,
                       step: int, seed: int) -> tuple[Optional[float], Optional[int]]:
    """Pick a same-step P from an independent calibration table.

    The same-step cyclic shuffle preserves each timestep's marginal P
    distribution while breaking the target prompt/state association. This is a
    closed-loop deployable control because all values come from a precomputed
    table, not from the current trajectory's future.
    """
    if table is None:
        return None, None
    values = table.get("values", {})
    candidates: list[tuple[int, float]] = []
    for k, arr in values.items():
        try:
            pid = int(k)
        except ValueError:
            continue
        if not isinstance(arr, list) or step >= len(arr):
            continue
        try:
            candidates.append((pid, float(arr[step])))
        except (TypeError, ValueError):
            continue
    if not candidates:
        raise ValueError(f"p shuffle table has no values for step {step}")
    candidates.sort(key=lambda x: x[0])
    ids = [pid for pid, _ in candidates]
    if prompt_idx in ids:
        rank = ids.index(prompt_idx)
    else:
        rank = int(prompt_idx) % len(candidates)
    src_rank = (rank + _stable_offset(seed, step, len(candidates))) % len(candidates)
    src_pid, val = candidates[src_rank]
    return val, src_pid


def _prefix_shadow_observe(
    self,
    *,
    ori_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    depths: Tuple[int, ...],
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    controlnet_blocks_repeat: bool = False,
) -> Tuple[Dict[str, Any], Dict[int, torch.Tensor]]:
    """Read-only prefix observer used by the costed shadow gate.

    It returns scalar fields for prediction plus the current prefix residuals,
    but does not mutate persistent anchors. Anchor refreshes are taken from the
    normal full forward path so forced-full steps do not pay extra observer
    compute just to keep shadow memory aligned with E0 collection semantics.
    """
    if not depths:
        return {}, {}
    max_depth = max(depths)
    n_double = len(self.transformer_blocks)
    n_single = len(self.single_transformer_blocks)
    max_allowed = n_double + n_single
    if max_depth > max_allowed:
        raise RuntimeError(f"shadow depth {max_depth} exceeds FLUX block count {max_allowed}")

    anchors: Dict[int, torch.Tensor] = getattr(self, "gt_shadow_anchor_residuals", {})
    current: Dict[int, torch.Tensor] = {}
    fields: Dict[str, Any] = {
        "shadow_depths": ",".join(str(d) for d in depths),
        "shadow_emit_metrics": True,
        "shadow_update_anchor": False,
    }
    wanted = set(int(d) for d in depths)
    hidden = ori_hidden_states.detach().clone()
    enc = encoder_hidden_states.detach().clone()
    t0 = time.perf_counter()

    def _record(depth: int, h: torch.Tensor) -> None:
        residual = (h - ori_hidden_states).detach()
        current[depth] = residual
        prefix = f"online_shadow_d{depth:02d}"
        anchor = anchors.get(depth)
        fields[f"{prefix}_residual_norm"] = _norm(residual)
        fields[f"{prefix}_anchor_present"] = bool(anchor is not None and anchor.shape == residual.shape)
        fields[f"{prefix}_elapsed_ms"] = float((time.perf_counter() - t0) * 1000.0)
        if anchor is not None and anchor.shape == residual.shape:
            diff = residual - anchor
            fields[f"{prefix}_residual_to_anchor"] = _norm(diff)
            fields[f"{prefix}_residual_to_anchor_rel"] = _norm(diff) / (_norm(anchor) + EPS)
            fields[f"{prefix}_cos_anchor"] = _cos(residual, anchor)
            fields[f"{prefix}_anchor_residual_norm"] = _norm(anchor)
        else:
            fields[f"{prefix}_residual_to_anchor"] = None
            fields[f"{prefix}_residual_to_anchor_rel"] = None
            fields[f"{prefix}_cos_anchor"] = None
            fields[f"{prefix}_anchor_residual_norm"] = _norm(anchor) if anchor is not None else None

    for depth in range(1, max_depth + 1):
        if depth <= n_double:
            index_block = depth - 1
            block = self.transformer_blocks[index_block]
            enc, hidden = block(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(n_double / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden = hidden + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden = hidden + controlnet_block_samples[index_block // interval_control]
        else:
            index_block = depth - n_double - 1
            block = self.single_transformer_blocks[index_block]
            enc, hidden = block(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(n_single / len(controlnet_single_block_samples)))
                hidden = hidden + controlnet_single_block_samples[index_block // interval_control]
        if depth in wanted:
            _record(depth, hidden)

    fields["online_shadow_total_elapsed_ms"] = float((time.perf_counter() - t0) * 1000.0)
    return fields, current


def _ffro_velocity_head_fields(
    transformer,
    *,
    history_state: Optional[Dict[str, Any]],
    step: int,
    ori_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step_size_abs: Optional[float],
) -> Dict[str, Any]:
    """Velocity-head FFRO scores without running transformer blocks."""

    out: Dict[str, Any] = {
        "online_ffro_vel_available_pre": False,
        "online_ffro_reuse_velocity_norm_pre": None,
        "online_ffro_taylor_o1_vel_norm_pre": None,
        "online_ffro_taylor_o1_step_norm_pre": None,
        "online_ffro_taylor_o2_vel_norm_pre": None,
        "online_ffro_taylor_o2_step_norm_pre": None,
        "online_ffro_hicache_o2_vel_norm_pre": None,
        "online_ffro_hicache_o2_step_norm_pre": None,
        "online_ffro_ensemble_vel_min_pre": None,
        "online_ffro_ensemble_vel_mean_pre": None,
        "online_ffro_ensemble_vel_max_pre": None,
        "online_ffro_ensemble_step_min_pre": None,
        "online_ffro_ensemble_step_mean_pre": None,
        "online_ffro_ensemble_step_max_pre": None,
        "online_ffro_vel_elapsed_ms_pre": 0.0,
    }
    preds = history_fd_forecast_predictions(
        history_state,
        step=int(step),
        sigma=float(getattr(transformer, "gt_history_fd_sigma", 0.5)),
    )
    reuse = preds.get("reuse")
    if reuse is None:
        return out
    start = time.perf_counter()
    with torch.no_grad():
        v_reuse = transformer.proj_out(transformer.norm_out(ori_hidden_states + reuse, temb))
        out["online_ffro_vel_available_pre"] = True
        out["online_ffro_reuse_velocity_norm_pre"] = _norm(v_reuse)
        vals: List[float] = []
        for name, suffix in (
            ("taylor_o1", "taylor_o1"),
            ("taylor_o2", "taylor_o2"),
            ("hicache_o2", "hicache_o2"),
        ):
            pred = preds.get(name)
            if pred is None or pred.shape != reuse.shape:
                out[f"online_ffro_{suffix}_vel_norm_pre"] = None
                out[f"online_ffro_{suffix}_step_norm_pre"] = None
                continue
            v_pred = transformer.proj_out(transformer.norm_out(ori_hidden_states + pred, temb))
            defect = _norm(v_reuse.detach().to(torch.float32) - v_pred.detach().to(torch.float32))
            out[f"online_ffro_{suffix}_vel_norm_pre"] = defect
            out[f"online_ffro_{suffix}_step_norm_pre"] = (
                None if step_size_abs is None else float(step_size_abs) * defect
            )
            vals.append(defect)
        if vals:
            out["online_ffro_ensemble_vel_min_pre"] = min(vals)
            out["online_ffro_ensemble_vel_mean_pre"] = float(sum(vals) / len(vals))
            out["online_ffro_ensemble_vel_max_pre"] = max(vals)
            if step_size_abs is not None:
                out["online_ffro_ensemble_step_min_pre"] = float(step_size_abs) * min(vals)
                out["online_ffro_ensemble_step_mean_pre"] = float(step_size_abs) * float(sum(vals) / len(vals))
                out["online_ffro_ensemble_step_max_pre"] = float(step_size_abs) * max(vals)
        out["online_ffro_vel_elapsed_ms_pre"] = (time.perf_counter() - start) * 1000.0
        return out


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------
def _gate_forward(
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

    sencache_latent_for_gate = hidden_states.detach()
    sencache_timestep_for_gate = float(timestep.detach().to(torch.float32).reshape(-1)[0].item() * 1000.0)
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

    cnt = int(self.gt_cnt)
    N = int(self.gt_num_steps)
    ori_hidden_states = hidden_states

    # ---- gate feature psi (SEA-filtered modulated first-block input) -----
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
    psi_drift = (rel_l1(psi, self.gt_prev_psi)
                 if self.gt_prev_psi is not None else 0.0)
    psi_disp = (rel_l1(psi, self.gt_anchor_psi)
                if self.gt_anchor_psi is not None else 0.0)
    self.gt_prev_psi = psi.detach()

    p_path_pre = float(getattr(self.gt_tracker, "_p_acc", 0.0)) + float(psi_drift)
    if self.gt_p_mode == "displacement":
        p_effective = float(psi_disp)
    elif self.gt_p_mode == "mix":
        p_effective = _p_mix(
            p_path_pre,
            float(psi_disp),
            float(getattr(self, "gt_p_mix_beta", 0.5)),
        )
    else:
        p_effective = None
    feats = self.gt_tracker.observe(cnt, psi_drift, p_effective=p_effective)
    p_raw = float(feats["p_eff"])
    s_factor = float(self.gt_tracker.s_cal[cnt])
    h_abs = float(self.gt_tracker.q_abs[cnt])
    a_factor = float(self.gt_tracker.a_cal[cnt])
    v_factor = s_factor * h_abs * a_factor
    sencache_fields = sencache_online_fields(
        table=getattr(self, "gt_sencache_table", None),
        current_latent=sencache_latent_for_gate,
        current_timestep=sencache_timestep_for_gate,
        anchor_latent=getattr(self, "gt_sencache_anchor_latent", None),
        anchor_timestep=getattr(self, "gt_sencache_anchor_timestep", None),
        anchor_step=getattr(self, "gt_sencache_anchor_step", None),
    )
    sencache_score = sencache_fields.get("online_sencache_score_pre")
    p_shuffle_value, p_shuffle_source_prompt_idx = (None, None)
    if self.gt_q_variant == "shuffled_p":
        p_shuffle_value, p_shuffle_source_prompt_idx = _lookup_shuffled_p(
            self.gt_p_shuffle_table,
            prompt_idx=int(self.gt_current_prompt_idx),
            step=cnt,
            seed=int(self.gt_p_shuffle_seed),
        )
    p_score = (
        1.0 if self.gt_q_variant == "v_only"
        else (None if self.gt_q_variant == "senqa" and sencache_score is None
              else (float(sencache_score) if self.gt_q_variant == "senqa"
                    else (float(p_shuffle_value) if self.gt_q_variant == "shuffled_p" else p_raw)))
    )
    p_source = ("none" if self.gt_q_variant == "v_only"
                else ("sencache_table" if self.gt_q_variant == "senqa"
                      else ("shuffle_table" if self.gt_q_variant == "shuffled_p" else "current")))
    q_path = float(feats["p_acc"]) * v_factor
    q_disp = float(psi_disp) * v_factor
    p_mix_value = _p_mix(
        float(feats["p_acc"]),
        float(psi_disp),
        float(getattr(self, "gt_p_mix_beta", 0.5)),
    )
    q_mix = float(p_mix_value) * v_factor
    score_full = p_raw * v_factor
    if self.gt_q_variant == "senqa" and sencache_score is None:
        score_variant = None
    else:
        score_variant = _q_variant_score(
            str(self.gt_q_variant),
            p=p_raw,
            s=s_factor,
            h=h_abs,
            a=a_factor,
            sen=sencache_score,
            p_shuf=p_shuffle_value,
        )
    # ---- E2 proxy-family features for proxy_ridge / stale_ridge / shadow_ridge
    proxy_prev = getattr(self, "gt_proxy_prev_modulated_input", None)
    proxy_prev_residual = getattr(self, "gt_previous_residual", None)
    proxy_feature_current = modulated_inp
    proxy_inc_raw = None
    proxy_inc_rescaled = None
    if proxy_prev is not None:
        if str(self.gt_proxy_mode) == "SeaCache":
            proxy_feature_current = modulated_inp.reshape(
                modulated_inp.shape[0],
                int(img_ids[:, 1].max().item() + 1),
                int(img_ids[:, 2].max().item() + 1),
                modulated_inp.shape[-1],
            )
            proxy_feature_current = apply_sea_with_scheduler(
                proxy_feature_current,
                self.scheduler,
                cnt,
                power_exp=2.0,
                dims=(-2, -3),
                norm_mode="mean",
            )
            proxy_feature_current = proxy_feature_current.reshape(
                proxy_feature_current.shape[0], -1, proxy_feature_current.shape[-1])
            proxy_inc_raw = rel_l1(proxy_feature_current, proxy_prev)
            proxy_inc_rescaled = proxy_inc_raw
        elif str(self.gt_proxy_mode) == "TeaCache":
            proxy_inc_raw = rel_l1(proxy_feature_current, proxy_prev)
            proxy_inc_rescaled = float(self.gt_teacache_rescale(proxy_inc_raw))
        else:
            raise RuntimeError(f"unknown proxy_mode: {self.gt_proxy_mode!r}")

    proxy_acc_before = float(getattr(self, "gt_proxy_accumulator", 0.0))
    proxy_acc_after_increment = (
        proxy_acc_before + float(proxy_inc_rescaled)
        if proxy_inc_rescaled is not None
        else None
    )
    proxy_threshold = float(getattr(self, "gt_proxy_cache_threshold", 0.0))
    proxy_p_path_decision = proxy_acc_after_increment
    proxy_q_no_sa = (
        float(proxy_p_path_decision) * h_abs
        if proxy_p_path_decision is not None
        else None
    )
    proxy_cache_age = cnt - int(getattr(self, "gt_proxy_last_refresh_step", 0))
    history_fd_state_pre = clone_history_fd_state(getattr(self, "gt_history_fd_state", None))
    history_fd_fields = history_fd_online_fields(
        history_fd_state_pre,
        step=cnt,
        sigma=float(getattr(self, "gt_history_fd_sigma", 0.5)),
    )
    proxy_family = str((getattr(self, "gt_proxy_model", None) or {}).get("family", ""))
    formula_variant = str(getattr(self, "gt_formula_variant", "q_native"))
    ffro_fields: Dict[str, Any] = {}
    if proxy_family.startswith("ffro") or (self.gt_kind == "formula" and formula_variant in FFRO_FORMULA_VARIANTS):
        ffro_fields.update(history_fd_ffro_residual_fields(
            history_fd_state_pre,
            step=cnt,
            sigma=float(getattr(self, "gt_history_fd_sigma", 0.5)),
        ))
        if (
            proxy_family.startswith("ffro")
            or formula_variant in ("ffro_step", "ffro_gv", "q_ffro_gv_veto")
        ):
            ffro_fields.update(_ffro_velocity_head_fields(
                self,
                history_state=history_fd_state_pre,
                step=cnt,
                ori_hidden_states=ori_hidden_states,
                temb=temb,
                step_size_abs=float(h_abs),
            ))
    proxy_row: Dict[str, Any] = {
        "mode": str(self.gt_proxy_mode),
        "cache_threshold": float(proxy_threshold),
        "fork_step": int(cnt),
        "step_index": int(cnt),
        "online_last_refresh_step_pre": int(getattr(self, "gt_proxy_last_refresh_step", 0)),
        "online_cache_age_pre": int(proxy_cache_age),
        "online_cache_run_len_pre": int(getattr(self, "gt_proxy_cache_run_len", 0)),
        "online_num_cached_so_far_pre": int(getattr(self, "gt_proxy_num_cached_so_far", 0)),
        "online_cache_ratio_so_far_pre": float(
            int(getattr(self, "gt_proxy_num_cached_so_far", 0)) / max(cnt, 1)
        ),
        "online_c_stale_pre": float(getattr(self, "gt_proxy_c_stale", 0.0)),
        "online_c_traj_pre": float(getattr(self, "gt_proxy_c_traj", 0.0)),
        "online_c_mem_pre": float(getattr(self, "gt_proxy_c_mem", 0.0)),
        "online_step_size_abs": float(h_abs),
        "online_gate_increment_raw": proxy_inc_raw,
        "online_gate_increment_rescaled": proxy_inc_rescaled,
        "online_gate_accumulator_before": proxy_acc_before,
        "online_threshold_margin_before": proxy_threshold - proxy_acc_before,
        "online_threshold_margin_after": (
            proxy_threshold - float(proxy_acc_after_increment)
            if proxy_acc_after_increment is not None
            else None
        ),
        "online_p_path_decision": proxy_p_path_decision,
        "online_q_proxy_no_sa": proxy_q_no_sa,
        "online_previous_modulated_input_norm": _norm_optional(proxy_prev),
        "online_previous_residual_norm": _norm_optional(proxy_prev_residual),
        **sencache_fields,
        **history_fd_fields,
        **ffro_fields,
    }
    formula_fields: Dict[str, Any] = {}
    if self.gt_kind == "formula":
        formula_fields = _formula_scores(
            formula_variant,
            proxy_row=proxy_row,
            score_full=float(score_full),
            s_factor=float(s_factor),
            h_abs=float(h_abs),
            a_factor=float(a_factor),
            formula_table=getattr(self, "gt_formula_table", None),
        )
    # ---- force-full constraints -------------------------------------------
    force_full_reason = None
    if cnt == 0:
        force_full_reason = "step0"
    elif cnt == N - 1:
        force_full_reason = "final_step"
    elif cnt < int(self.gt_first_enhance):
        force_full_reason = "first_enhance"
    elif self.gt_previous_residual is None:
        force_full_reason = "no_previous_residual"
    elif (
        (
            self.gt_q_variant == "senqa"
            or str((getattr(self, "gt_proxy_model", None) or {}).get("family", "")).startswith("sencache_")
        )
        and getattr(self, "gt_sencache_anchor_latent", None) is None
    ):
        force_full_reason = "no_sencache_anchor"
    elif self.gt_kind == "formula" and not bool(formula_fields.get("formula_available")):
        force_full_reason = "formula_unavailable"
    elif getattr(self, "gt_action_steps", set()) and cnt not in self.gt_action_steps:
        force_full_reason = "not_in_action_steps"
    force_full = force_full_reason is not None

    shadow_depths = tuple(int(d) for d in getattr(self, "gt_shadow_depths", ()) or ())
    shadow_anchor_step_pre = getattr(self, "gt_shadow_anchor_step", None)
    shadow_fields: Dict[str, Any] = {}
    shadow_prediction_eligible = bool(
        self.gt_kind == "shadow_ridge" and bool(shadow_depths) and not force_full
    )
    if self.gt_kind != "shadow_ridge":
        shadow_observed_reason = "skipped_not_shadow_gate"
    elif not shadow_depths:
        shadow_observed_reason = "skipped_no_shadow_depths"
    elif force_full_reason is not None:
        shadow_observed_reason = f"skipped_forced_full_{force_full_reason}"
    else:
        shadow_observed_reason = "observed_for_prediction"

    if shadow_prediction_eligible:
        shadow_fields, _ = _prefix_shadow_observe(
            self,
            ori_hidden_states=ori_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            depths=shadow_depths,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
        proxy_row.update(shadow_fields)

    # ---- decision -------------------------------------------------------
    # Adaptive τ (negative-feedback budget tracker) — only active when
    # gt_target_cached_ratio > 0 AND gt_adapt_rate > 0; otherwise tau_eff
    # collapses to the static threshold, reproducing the original q/est2 gate.
    #
    # Two safeguards (added after a smoke run revealed step-1 catastrophe):
    #   1. Burn-in: skip adaptation until cnt >= adapt_burn_in. At cnt small,
    #      n_proc is tiny and `(n_cached/n_proc) − target` swings by ±target,
    #      blowing τ_eff up by exp(rate * target) ~ 100×+ and forcing the
    #      gate to cache the highest-q early step (catastrophic for PSNR).
    #   2. Overshoot clip: |overshoot| ≤ adapt_overshoot_cap to keep
    #      exp(rate · overshoot) bounded.
    if (self.gt_target_cached_ratio > 0 and self.gt_adapt_rate > 0
            and cnt >= int(self.gt_adapt_burn_in)):
        n_proc = cnt  # past decisions already committed
        current_ratio = float(self.gt_n_cached_so_far) / n_proc
        overshoot = current_ratio - float(self.gt_target_cached_ratio)
        cap = float(self.gt_adapt_overshoot_cap)
        overshoot_clip = max(-cap, min(cap, overshoot))
        tau_eff = float(self.gt_threshold) * math.exp(
            -float(self.gt_adapt_rate) * overshoot_clip)
    else:
        overshoot = 0.0
        tau_eff = float(self.gt_threshold)

    if force_full:
        u_n = 0
        predict = 0.0
    elif self.gt_kind == "interval":
        u_n = 1 if (cnt % int(self.gt_interval) != 0) else 0
        predict = 0.0
    elif self.gt_kind == "q":
        predict = float(score_variant) if score_variant is not None else 1e300
        u_n = 1 if (predict < tau_eff
                    and feats["gap"] < int(self.gt_g_max)) else 0
    elif self.gt_kind == "est2":
        predict = _est2_predict(feats["q"], feats["gap"], cnt,
                                feats["c_stale"], feats["c_traj"], feats["c_mem"],
                                self.gt_beta)
        u_n = 1 if (predict < tau_eff
                    and feats["gap"] < int(self.gt_g_max)) else 0
    elif self.gt_kind == "formula":
        predict = float(formula_fields["formula_predict"]) if formula_fields.get("formula_available") else 1e300
        u_n = 1 if (predict < tau_eff
                    and int(proxy_row["online_cache_run_len_pre"]) < int(self.gt_g_max)) else 0
    elif self.gt_kind in ("proxy_ridge", "stale_ridge", "shadow_ridge"):
        model = getattr(self, "gt_proxy_model", None)
        if model is None:
            raise RuntimeError(f"{self.gt_kind} requires gt_proxy_model")
        predict = _proxy_predict_log1p(model, proxy_row)
        u_n = 1 if (predict < tau_eff
                    and int(proxy_row["online_cache_run_len_pre"]) < int(self.gt_g_max)) else 0
    else:
        raise ValueError(f"unknown gate kind: {self.gt_kind!r}")

    self.gt_tracker.commit(cnt, u_n)
    if u_n == 0:
        self.gt_anchor_psi = psi.detach()
        self.gt_anchor_step = cnt
    last_refresh_logged = cnt if u_n == 0 else int(cnt - int(feats["gap"]))
    if u_n == 1:
        self.gt_n_cached_so_far += 1
    self.gt_proxy_prev_modulated_input = proxy_feature_current.detach()
    old_proxy_traj = float(getattr(self, "gt_proxy_c_traj", 0.0))
    proxy_commit_q = float(proxy_q_no_sa or 0.0)
    if u_n == 1:
        self.gt_proxy_accumulator = float(proxy_acc_after_increment or proxy_acc_before)
        self.gt_proxy_c_traj = old_proxy_traj + proxy_commit_q
        self.gt_proxy_c_stale = float(getattr(self, "gt_proxy_c_stale", 0.0)) + proxy_commit_q
        self.gt_proxy_cache_run_len = int(getattr(self, "gt_proxy_cache_run_len", 0)) + 1
        self.gt_proxy_num_cached_so_far = int(getattr(self, "gt_proxy_num_cached_so_far", 0)) + 1
    else:
        self.gt_proxy_accumulator = 0.0
        self.gt_proxy_c_traj = old_proxy_traj
        self.gt_proxy_c_stale = 0.0
        self.gt_proxy_c_mem = old_proxy_traj
        self.gt_proxy_cache_run_len = 0
        self.gt_proxy_last_refresh_step = cnt
    decision_row = {
        "step": cnt, "u": int(u_n), "predict": float(predict),
        "q": float(feats["q"]), "gap": int(feats["gap"]),
        "proxy_family": (None if getattr(self, "gt_proxy_model", None) is None
                         else str(self.gt_proxy_model.get("family"))),
        "proxy_model_class": (None if getattr(self, "gt_proxy_model", None) is None
                              else str(self.gt_proxy_model.get("model_class", "ridge"))),
        "proxy_target": (None if getattr(self, "gt_proxy_model", None) is None
                         else str(self.gt_proxy_model.get("target"))),
        "proxy_model_path": getattr(self, "gt_proxy_model_path", None),
        "proxy_mode": str(self.gt_proxy_mode),
        "proxy_cache_threshold": float(proxy_threshold),
        "proxy_predict_log1p_risk": (
            float(predict)
            if self.gt_kind in ("proxy_ridge", "stale_ridge", "shadow_ridge") and not force_full
            else None
        ),
        **formula_fields,
        "formula_table_path": getattr(self, "gt_formula_table_path", None),
        "p_mode": str(self.gt_p_mode),
        "p_mix_beta": float(getattr(self, "gt_p_mix_beta", 0.5)),
        "anchor_step": (None if self.gt_anchor_step is None else int(self.gt_anchor_step)),
        "last_refresh": int(last_refresh_logged),
        "p_acc": float(feats["p_acc"]), "psi_drift": float(feats["psi_drift"]),
        "p_eff": float(feats["p_eff"]), "psi_disp": float(psi_disp),
        "p_path": float(feats["p_acc"]),
        "p_disp": float(psi_disp),
        "p_mix": float(p_mix_value),
        "q_variant": str(self.gt_q_variant),
        "score_variant": str(self.gt_q_variant),
        "score_full": float(score_full),
        "score_variant_value": None if score_variant is None else float(score_variant),
        "p_raw": float(p_raw),
        "p_score": None if p_score is None else float(p_score),
        "p_source": str(p_source),
        "sencache_score": None if sencache_score is None else float(sencache_score),
        "sencache_sensitivity_sha256": getattr(self, "gt_sencache_table_sha256", None),
        "q_senqa": None if sencache_score is None else float(float(sencache_score) * h_abs * a_factor),
        "s_factor": float(s_factor),
        "h_abs": float(h_abs),
        "a_factor": float(a_factor),
        "v_factor": float(v_factor),
        **_q_variant_uses(str(self.gt_q_variant)),
        "force_full": bool(force_full),
        "force_full_reason": force_full_reason,
        "p_shuffle_seed": (None if self.gt_q_variant != "shuffled_p"
                           else int(self.gt_p_shuffle_seed)),
        "p_shuffle_scope": (None if self.gt_q_variant != "shuffled_p"
                            else str(self.gt_p_shuffle_scope)),
        "p_shuffle_source_prompt_idx": p_shuffle_source_prompt_idx,
        "q_path": float(q_path),
        "q_disp": float(q_disp),
        "q_mix": float(q_mix),
        "c_stale": float(feats["c_stale"]),
        "c_traj": float(feats["c_traj"]),
        "c_mem": float(feats["c_mem"]),
        **{k: (bool(v) if isinstance(v, bool)
               else (int(v) if isinstance(v, int)
                     else (float(v) if isinstance(v, float) and v is not None else v)))
           for k, v in proxy_row.items()},
        "shadow_extra_compute": bool(self.gt_kind == "shadow_ridge"),
        "shadow_observer_extra_compute": bool(shadow_fields),
        "shadow_depths": [int(d) for d in shadow_depths],
        "online_shadow_total_elapsed_ms": shadow_fields.get("online_shadow_total_elapsed_ms"),
        "shadow_observer_ran": bool(shadow_fields),
        "shadow_emit_metrics": bool(shadow_fields),
        "shadow_observed_reason": shadow_observed_reason,
        "shadow_eligible_for_prediction": bool(shadow_prediction_eligible),
        "shadow_anchor_step_pre": (
            None if shadow_anchor_step_pre is None else int(shadow_anchor_step_pre)
        ),
        "tau_eff": float(tau_eff),
        "overshoot": float(overshoot),
    }

    # ---- cache (zero-order whole-transformer residual reuse) or full ----
    shadow_anchor_update_needed = bool(
        self.gt_kind == "shadow_ridge" and u_n == 0 and bool(shadow_depths)
    )
    shadow_anchor_updated = False
    shadow_anchor_update_source = None
    shadow_full_anchors: Dict[int, torch.Tensor] = {}
    shadow_wanted_depths = set(shadow_depths) if shadow_anchor_update_needed else set()
    if shadow_anchor_update_needed:
        max_allowed_shadow_depth = len(self.transformer_blocks) + len(self.single_transformer_blocks)
        max_shadow_depth = max(shadow_wanted_depths)
        if max_shadow_depth > max_allowed_shadow_depth:
            raise RuntimeError(
                f"shadow depth {max_shadow_depth} exceeds FLUX block count {max_allowed_shadow_depth}"
            )

    payload_fields: Dict[str, Any] = {
        "payload_mode": str(getattr(self, "gt_payload_mode", "reuse")),
        "payload_blend": float(getattr(self, "gt_payload_blend", 1.0)),
        "payload_used": None,
        "payload_available": None,
        "payload_fallback": None,
        "payload_reuse_norm": None,
        "payload_forecast_norm": None,
        "payload_delta_from_reuse_norm": None,
    }
    if u_n == 1 and self.gt_previous_residual is not None:
        reuse_payload = self.gt_previous_residual
        forecast_payload, forecast_fields = _forecast_payload(
            str(getattr(self, "gt_payload_mode", "reuse")),
            reuse=reuse_payload,
            history_state=history_fd_state_pre,
            step=cnt,
            sigma=float(getattr(self, "gt_history_fd_sigma", 0.5)),
        )
        blend = float(getattr(self, "gt_payload_blend", 1.0))
        if forecast_fields["payload_used"] == "reuse" or str(getattr(self, "gt_payload_mode", "reuse")) == "reuse":
            cache_payload = reuse_payload
        else:
            cache_payload = (1.0 - blend) * reuse_payload + blend * forecast_payload
        payload_fields.update(forecast_fields)
        payload_fields["payload_blend"] = blend
        if forecast_fields["payload_used"] != "reuse" and blend < 1.0:
            payload_fields["payload_used"] = f"blend_{forecast_fields['payload_used']}"
            payload_fields["payload_delta_from_reuse_norm"] = _norm(
                cache_payload.detach().to(torch.float32) - reuse_payload.detach().to(torch.float32)
            )
        hidden_states = hidden_states + cache_payload
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
            depth = index_block + 1
            if depth in shadow_wanted_depths:
                shadow_full_anchors[depth] = (hidden_states - ori_hidden_states).detach()
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
            depth = len(self.transformer_blocks) + index_block + 1
            if depth in shadow_wanted_depths:
                shadow_full_anchors[depth] = (hidden_states - ori_hidden_states).detach()
        self.gt_previous_residual = hidden_states - ori_hidden_states
        self.gt_history_fd_state = history_fd_update_on_full(
            history_fd_state_pre,
            residual=self.gt_previous_residual,
            step=cnt,
            sigma=float(getattr(self, "gt_history_fd_sigma", 0.5)),
            ema_beta=float(getattr(self, "gt_history_fd_ema_beta", 0.2)),
        )
        if getattr(self, "gt_sencache_table", None) is not None:
            self.gt_sencache_anchor_latent = sencache_latent_for_gate.detach().clone()
            self.gt_sencache_anchor_timestep = float(sencache_timestep_for_gate)
            self.gt_sencache_anchor_step = int(cnt)
        if shadow_anchor_update_needed:
            missing_depths = sorted(shadow_wanted_depths - set(shadow_full_anchors))
            if missing_depths:
                raise RuntimeError(f"shadow full-forward anchor missing depths: {missing_depths}")
            self.gt_shadow_anchor_residuals = {
                depth: tensor.detach()
                for depth, tensor in shadow_full_anchors.items()
            }
            self.gt_shadow_anchor_step = cnt
            shadow_anchor_updated = True
            shadow_anchor_update_source = "full_forward"

    shadow_anchor_step_post = getattr(self, "gt_shadow_anchor_step", None)
    decision_row.update({
        **payload_fields,
        "shadow_update_anchor": bool(shadow_anchor_updated),
        "shadow_anchor_updated": bool(shadow_anchor_updated),
        "shadow_anchor_update_source": shadow_anchor_update_source,
        "shadow_anchor_step_post": (
            None if shadow_anchor_step_post is None else int(shadow_anchor_step_post)
        ),
    })
    self.gt_decisions.append(decision_row)

    self.gt_cnt += 1
    if self.gt_cnt == N:
        self.gt_cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install_gate(pipe, *, kind: str, threshold: float, g_max: int,
                 interval: int, first_enhance: int, num_steps: int,
                 q_abs, s_cal, a_cal, beta: Optional[List[float]] = None,
                 p_mode: str = "path",
                 q_variant: str = "full",
                 formula_variant: str = "q_native",
                 formula_table: Optional[Dict[str, Any]] = None,
                 formula_table_path: Optional[str] = None,
                 p_shuffle_table: Optional[Dict[str, Any]] = None,
                 p_shuffle_seed: int = 12345,
                 p_shuffle_scope: str = "same_step_prompt",
                 mem_variant: int = 1,
                 target_cached_ratio: float = 0.0,
                 adapt_rate: float = 0.0,
                 adapt_burn_in: int = 10,
                 adapt_overshoot_cap: float = 0.1,
                 proxy_model: Optional[Dict[str, Any]] = None,
                 proxy_model_path: Optional[str] = None,
                 proxy_mode: str = "SeaCache",
                 proxy_cache_threshold: float = 0.3,
                 sencache_sensitivity_path: Optional[str] = None,
                 teacache_backbone: str = "flux",
                 teacache_variant: Optional[str] = None,
                 p_mix_beta: float = 0.5,
                 payload_mode: str = "reuse",
                 payload_blend: float = 1.0,
                 shadow_depths: Tuple[int, ...] = (),
                 action_steps: Optional[set[int]] = None) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward with the SM-D gate.

    When both `target_cached_ratio > 0` and `adapt_rate > 0`, the gate
    applies a negative-feedback τ adjustment online:

        τ_eff(n) = threshold · exp(−adapt_rate · (running_cached_ratio
                                                  − target_cached_ratio))

    so caching slows down when the running ratio exceeds target. With
    either knob ≤ 0 the gate reduces to the static-threshold behaviour.
    """
    if kind not in ("interval", "q", "est2", "formula", "proxy_ridge", "stale_ridge", "shadow_ridge"):
        raise ValueError(f"unknown gate kind: {kind!r}")
    if p_mode not in P_MODES:
        raise ValueError(f"unknown p_mode: {p_mode!r}")
    if not (0.0 <= float(p_mix_beta) <= 1.0):
        raise ValueError(f"p_mix_beta must be in [0, 1], got {p_mix_beta}")
    if q_variant not in Q_VARIANTS:
        raise ValueError(f"unknown q_variant: {q_variant!r}")
    if payload_mode not in PAYLOAD_MODES:
        raise ValueError(f"unknown payload_mode: {payload_mode!r}")
    if not (0.0 <= float(payload_blend) <= 1.0):
        raise ValueError(f"payload_blend must be in [0, 1], got {payload_blend}")
    if formula_variant not in FORMULA_VARIANTS:
        raise ValueError(f"unknown formula_variant: {formula_variant!r}")
    if kind != "formula" and formula_variant != "q_native":
        raise ValueError("--formula_variant is only meaningful for --gate formula")
    if kind == "formula" and formula_variant in PERCENTILE_FORMULA_VARIANTS and formula_table is None:
        raise ValueError(f"--formula_variant {formula_variant} needs --formula_table")
    if kind == "est2" and (beta is None or len(beta) != 8):
        raise ValueError("est2 gate needs an 8-element beta")
    if kind == "est2" and p_mode != "path":
        raise ValueError("est2 coefficients were trained with p_mode='path'")
    if kind == "est2" and q_variant != "full":
        raise ValueError("est2 coefficients were trained with q_variant='full'")
    if kind == "stale_ridge":
        if proxy_model is None or proxy_model.get("family") != "stale_state":
            raise ValueError("stale_ridge gate needs a stale_state --proxy_model")
    if kind == "proxy_ridge":
        if proxy_model is None or proxy_model.get("family") not in (
            "step_gap", "q_native", "stale_state", "fd_history",
            "ffro", "ffro_plus_stale_state",
            "ffro_res_reuse", "ffro_res_forecaster_t1", "ffro_res_forecaster_t2",
            "ffro_res_forecaster_h2", "ffro_res_spread", "ffro_res_ensemble",
            "ffro_vel_only", "ffro_step_only",
            "ffro_vel_plus_stale_state", "ffro_step_plus_stale_state",
            "sencache_local", "sencache_state",
        ):
            raise ValueError(
                "proxy_ridge gate needs a strict-online --proxy_model "
                "with family step_gap, q_native, stale_state, fd_history, "
                "ffro variants, sencache_local, or sencache_state"
            )
    if kind == "shadow_ridge":
        if proxy_model is None or proxy_model.get("family") != "shadow":
            raise ValueError("shadow_ridge gate needs a shadow --proxy_model")
        if not shadow_depths:
            raise ValueError("shadow_ridge gate needs --shadow_depths")
    if proxy_mode not in ("SeaCache", "TeaCache"):
        raise ValueError(f"unknown proxy_mode: {proxy_mode!r}")
    if q_variant == "shuffled_p" and p_shuffle_table is None:
        raise ValueError("q_variant='shuffled_p' needs --p_shuffle_table")
    needs_sencache = (
        q_variant == "senqa"
        or (
            proxy_model is not None
            and str(proxy_model.get("family", "")).startswith("sencache_")
        )
    )
    if needs_sencache and not sencache_sensitivity_path:
        raise ValueError("SenCache q/proxy variants need --sencache_sensitivity_path")
    if p_shuffle_scope != "same_step_prompt":
        raise ValueError("only p_shuffle_scope='same_step_prompt' is supported")
    if target_cached_ratio < 0 or target_cached_ratio >= 1:
        raise ValueError(f"target_cached_ratio must be in [0, 1); "
                         f"got {target_cached_ratio}")

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _gate_forward
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.gt_kind = str(kind)
    tr.gt_p_mode = str(p_mode)
    tr.gt_p_mix_beta = float(p_mix_beta)
    tr.gt_q_variant = str(q_variant)
    tr.gt_payload_mode = str(payload_mode)
    tr.gt_payload_blend = float(payload_blend)
    tr.gt_formula_variant = str(formula_variant)
    tr.gt_formula_table = formula_table
    tr.gt_formula_table_path = formula_table_path
    tr.gt_p_shuffle_table = p_shuffle_table
    tr.gt_p_shuffle_seed = int(p_shuffle_seed)
    tr.gt_p_shuffle_scope = str(p_shuffle_scope)
    tr.gt_threshold = float(threshold)
    tr.gt_g_max = int(g_max)
    tr.gt_interval = int(interval)
    tr.gt_first_enhance = int(first_enhance)
    tr.gt_num_steps = int(num_steps)
    tr.gt_beta = list(beta) if beta is not None else None
    tr.gt_target_cached_ratio = float(target_cached_ratio)
    tr.gt_adapt_rate = float(adapt_rate)
    tr.gt_adapt_burn_in = int(adapt_burn_in)
    tr.gt_adapt_overshoot_cap = float(adapt_overshoot_cap)
    tr.gt_proxy_model = proxy_model
    tr.gt_proxy_model_path = proxy_model_path
    tr.gt_proxy_mode = str(proxy_mode)
    tr.gt_proxy_cache_threshold = float(proxy_cache_threshold)
    table = load_sensitivity_table(sencache_sensitivity_path) if sencache_sensitivity_path else None
    tr.gt_sencache_table = table
    tr.gt_sencache_table_sha256 = table.sha256 if table is not None else None
    tr.gt_sencache_table_metadata = dict(table.metadata) if table is not None else None
    tr.gt_sencache_anchor_latent = None
    tr.gt_sencache_anchor_timestep = None
    tr.gt_sencache_anchor_step = None
    tr.gt_teacache_rescale = np.poly1d(get_coeffs(teacache_backbone, teacache_variant))
    tr.gt_shadow_depths = tuple(int(d) for d in shadow_depths)
    tr.gt_shadow_anchor_residuals = {}
    tr.gt_shadow_anchor_step = None
    tr.gt_history_fd_state = init_history_fd_state()
    tr.gt_history_fd_sigma = 0.5
    tr.gt_history_fd_ema_beta = 0.2
    tr.gt_action_steps = set(action_steps or set())
    tr.gt_tracker = MarginalFeatureTracker(q_abs, s_cal, a_cal,
                                           mem_variant=int(mem_variant))
    tr.gt_cnt = 0
    tr.gt_prev_psi = None
    tr.gt_anchor_psi = None
    tr.gt_anchor_step = None
    tr.gt_previous_residual = None
    tr.gt_decisions = []
    tr.gt_n_cached_so_far = 0
    tr.gt_current_prompt_idx = 0
    tr.gt_proxy_prev_modulated_input = None
    tr.gt_proxy_accumulator = 0.0
    tr.gt_proxy_last_refresh_step = 0
    tr.gt_proxy_cache_run_len = 0
    tr.gt_proxy_num_cached_so_far = 0
    tr.gt_proxy_c_stale = 0.0
    tr.gt_proxy_c_traj = 0.0
    tr.gt_proxy_c_mem = 0.0
    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("scheduler", "gt_kind", "gt_threshold", "gt_g_max",
                     "gt_p_mode", "gt_q_variant", "gt_p_shuffle_table",
                     "gt_p_mix_beta", "gt_payload_mode", "gt_payload_blend",
                     "gt_formula_variant", "gt_formula_table", "gt_formula_table_path",
                     "gt_p_shuffle_seed", "gt_p_shuffle_scope",
                     "gt_interval", "gt_first_enhance", "gt_num_steps",
                     "gt_beta", "gt_target_cached_ratio", "gt_adapt_rate",
                     "gt_adapt_burn_in", "gt_adapt_overshoot_cap",
                     "gt_proxy_model", "gt_proxy_model_path", "gt_proxy_mode",
                     "gt_proxy_cache_threshold", "gt_sencache_table",
                     "gt_sencache_anchor_latent", "gt_sencache_anchor_timestep",
                     "gt_sencache_anchor_step", "gt_sencache_table_sha256",
                     "gt_sencache_table_metadata",
                     "gt_teacache_rescale",
                     "gt_shadow_depths", "gt_shadow_anchor_residuals",
                     "gt_shadow_anchor_step",
                     "gt_history_fd_state", "gt_history_fd_sigma",
                     "gt_history_fd_ema_beta",
                     "gt_action_steps",
                     "gt_tracker", "gt_cnt", "gt_prev_psi", "gt_anchor_psi",
                     "gt_anchor_step",
                     "gt_previous_residual", "gt_decisions",
                     "gt_n_cached_so_far", "gt_current_prompt_idx",
                     "gt_proxy_prev_modulated_input", "gt_proxy_accumulator",
                     "gt_proxy_last_refresh_step", "gt_proxy_cache_run_len",
                     "gt_proxy_num_cached_so_far", "gt_proxy_c_stale",
                     "gt_proxy_c_traj", "gt_proxy_c_mem"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_gate_state(pipe) -> None:
    """Per-trajectory reset before each pipe() call."""
    tr = pipe.transformer
    tr.gt_cnt = 0
    tr.gt_prev_psi = None
    tr.gt_anchor_psi = None
    tr.gt_anchor_step = None
    tr.gt_previous_residual = None
    tr.gt_decisions = []
    tr.gt_n_cached_so_far = 0
    tr.gt_proxy_prev_modulated_input = None
    tr.gt_proxy_accumulator = 0.0
    tr.gt_proxy_last_refresh_step = 0
    tr.gt_proxy_cache_run_len = 0
    tr.gt_proxy_num_cached_so_far = 0
    tr.gt_proxy_c_stale = 0.0
    tr.gt_proxy_c_traj = 0.0
    tr.gt_proxy_c_mem = 0.0
    tr.gt_sencache_anchor_latent = None
    tr.gt_sencache_anchor_timestep = None
    tr.gt_sencache_anchor_step = None
    tr.gt_shadow_anchor_residuals = {}
    tr.gt_shadow_anchor_step = None
    tr.gt_history_fd_state = init_history_fd_state()
    tr.gt_tracker.reset()


def _load_calib(q_k_path: Path, sa_calib_path: Path, num_steps: int):
    qd = json.loads(Path(q_k_path).read_text())
    q_abs = [abs(float(x)) for x in qd["Q_k"]]
    sd = json.loads(Path(sa_calib_path).read_text())
    s_cal = [float(x) for x in sd["S_k"]]
    a_cal = [float(x) for x in sd["A_k"]]
    for name, arr in (("Q_k", q_abs), ("S_k", s_cal), ("A_k", a_cal)):
        if len(arr) != num_steps:
            raise SystemExit(f"{name} length {len(arr)} != num_steps {num_steps}")
    return q_abs, s_cal, a_cal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp SM-D: closed-loop state-gate cache runner.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--gate", choices=["interval", "q", "est2", "formula", "proxy_ridge", "stale_ridge", "shadow_ridge"],
                   required=True)
    p.add_argument("--p_mode", choices=P_MODES, default="path",
                   help=("P factor used inside q_n. 'path' is the legacy "
                         "adjacent-drift accumulator; 'displacement' uses "
                         "rel_l1(current psi, last-full-anchor psi); 'mix' "
                         "uses a geometric path/displacement interpolation. "
                         "Only valid for --gate q."))
    p.add_argument("--p_mix_beta", type=float, default=0.5,
                   help=("For --p_mode mix, beta in [0,1] for "
                         "exp(beta*log(P_path)+(1-beta)*log(P_disp))."))
    p.add_argument("--q_variant", choices=Q_VARIANTS, default="full",
                   help=("Decision score for --gate q. 'full' is P*S*|h|*A; "
                         "other choices ablate factors while logging full q."))
    p.add_argument("--payload_mode", choices=PAYLOAD_MODES, default="reuse",
                   help=("Residual payload used on cached steps. 'reuse' is "
                         "SeaCache-style zero-order residual reuse; forecast "
                         "modes use only prior full-refresh residual history."))
    p.add_argument("--payload_blend", type=float, default=1.0,
                   help=("Blend weight for forecast payloads: payload = "
                         "(1-w)*reuse + w*forecast. Default 1.0."))
    p.add_argument("--formula_variant", choices=FORMULA_VARIANTS, default="q_native",
                   help=("Decision score for --gate formula. Veto variants use "
                         "per-step percentiles from --formula_table."))
    p.add_argument("--formula_table", type=Path, default=None,
                   help=("ffro_formula_percentile_table.v1 JSON built by "
                         "analysis/ffro_formula_audit.py; required for veto formulas."))
    p.add_argument("--p_shuffle_table", type=Path, default=None,
                   help=("JSON table built by analysis/build_qgate_p_shuffle_table.py; "
                         "required for --q_variant shuffled_p."))
    p.add_argument("--p_shuffle_seed", type=int, default=12345,
                   help="Deterministic same-step prompt shuffle seed.")
    p.add_argument("--p_shuffle_scope", choices=["same_step_prompt"],
                   default="same_step_prompt",
                   help="Shuffle control scope; only same_step_prompt is supported.")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="cache iff predict < threshold (q / est2 gates).")
    p.add_argument("--g_max", type=int, default=8,
                   help="hard cap on consecutive cached steps.")
    p.add_argument("--interval", type=int, default=2,
                   help="interval gate cadence (cache iff cnt %% interval != 0).")
    p.add_argument("--first_enhance", type=int, default=1)
    p.add_argument("--q_k", type=Path, required=True)
    p.add_argument("--sa_calib", type=Path, required=True)
    p.add_argument("--estimator_summary", type=Path, default=None,
                   help="path to SM-C marginal_estimator_summary.json (est2 gate).")
    p.add_argument("--proxy_model", type=Path, default=None,
                   help="online_proxy_gate_model.v1 JSON for proxy_ridge/stale_ridge/shadow_ridge.")
    p.add_argument("--proxy_mode", choices=["SeaCache", "TeaCache"], default="SeaCache",
                   help="Native q-like feature family used to compute deployment proxy fields.")
    p.add_argument("--proxy_cache_threshold", type=float, default=0.3,
                   help="Reference native cache threshold used only for proxy margin features.")
    p.add_argument("--sencache_sensitivity_path", type=Path, default=None,
                   help="Frozen FLUX SenCache sensitivity npz for q_variant=senqa or sencache_* proxy families.")
    p.add_argument("--teacache_backbone", default="flux")
    p.add_argument("--teacache_variant", default=None)
    p.add_argument("--shadow_depths", default="",
                   help="Comma/space-separated prefix depths for shadow_ridge, e.g. '1,2,4,8'.")
    p.add_argument("--action_steps", default="",
                   help=("Optional comma/space-separated steps where learned gates may cache. "
                         "Other steps are forced full; useful for E2-calibrated smoke runs."))
    p.add_argument("--allow_sparse_numeric_proxy", action="store_true",
                   help="Allow all-step deployment of a numeric proxy without dense all-step labels.")
    p.add_argument("--mem_variant", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--target_cached_ratio", type=float, default=0.0,
                   help=("Adaptive τ: target running cached_ratio in [0, 1). "
                         "0 disables (static threshold)."))
    p.add_argument("--adapt_rate", type=float, default=0.0,
                   help=("Adaptive τ: negative-feedback gain. τ_eff = τ · "
                         "exp(-adapt_rate · clip(running_ratio − target)). "
                         "0 disables. Sensible values 1.0 – 20.0."))
    p.add_argument("--adapt_burn_in", type=int, default=10,
                   help=("Steps before adaptation kicks in. At small cnt the "
                         "running_ratio − target swings ±target and explodes "
                         "τ_eff; burn-in keeps τ_eff at static τ until cnt ≥ "
                         "burn_in. Default 10."))
    p.add_argument("--adapt_overshoot_cap", type=float, default=0.1,
                   help=("|overshoot| ≤ cap before exp(). Bounds τ_eff swing "
                         "to exp(±rate·cap). Default 0.1."))
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
    return p.parse_args()


def _pipe_kwargs(prompt: str, seed: int, args) -> Dict[str, Any]:
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(int(seed))
    return dict(
        prompt=prompt,
        num_inference_steps=int(args.num_steps),
        guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    )


def main() -> int:
    args = parse_args()
    if args.gate in ("q", "est2", "formula", "proxy_ridge", "stale_ridge", "shadow_ridge") and args.threshold <= 0:
        raise SystemExit(
            f"--gate {args.gate} requires --threshold > 0 (got {args.threshold}); "
            "the decision score is positive/log-risk-like so threshold <= 0 caches little or nothing.")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    N = int(args.num_steps)
    q_abs, s_cal, a_cal = _load_calib(args.q_k, args.sa_calib, N)
    beta = None
    if args.gate == "est2":
        if args.p_mode != "path":
            raise SystemExit("--gate est2 only supports --p_mode path")
        if args.q_variant != "full":
            raise SystemExit("--gate est2 only supports --q_variant full")
        if args.estimator_summary is None:
            raise SystemExit("--gate est2 needs --estimator_summary")
        beta = _load_est_coef(args.estimator_summary, "est2")
    proxy_model = None
    if args.gate == "proxy_ridge":
        proxy_model = _load_proxy_model(
            args.proxy_model,
            allowed_families=(
                "step_gap", "q_native", "stale_state", "fd_history",
                "ffro", "ffro_plus_stale_state",
                "ffro_res_reuse", "ffro_res_forecaster_t1", "ffro_res_forecaster_t2",
                "ffro_res_forecaster_h2", "ffro_res_spread", "ffro_res_ensemble",
                "ffro_vel_only", "ffro_step_only",
                "ffro_vel_plus_stale_state", "ffro_step_plus_stale_state",
                "sencache_local", "sencache_state",
            ),
        )
    elif args.gate == "stale_ridge":
        proxy_model = _load_proxy_model(args.proxy_model, expected_family="stale_state")
    elif args.gate == "shadow_ridge":
        proxy_model = _load_proxy_model(args.proxy_model, expected_family="shadow")
        if not _parse_int_set(args.shadow_depths):
            raise SystemExit("--gate shadow_ridge needs --shadow_depths")
    if proxy_model is not None:
        _validate_proxy_step_support(
            proxy_model,
            action_steps=_parse_int_set(args.action_steps),
            num_steps=N,
            allow_sparse_numeric_proxy=bool(args.allow_sparse_numeric_proxy),
        )
        _validate_proxy_context(
            proxy_model,
            proxy_mode=str(args.proxy_mode),
            proxy_cache_threshold=float(args.proxy_cache_threshold),
        )
    if args.gate != "q" and args.q_variant != "full":
        raise SystemExit("--q_variant is only meaningful for --gate q")
    p_shuffle_table = _load_p_shuffle_table(args.p_shuffle_table)
    formula_table = _load_formula_table(args.formula_table)
    if args.q_variant == "shuffled_p" and p_shuffle_table is None:
        raise SystemExit("--q_variant shuffled_p needs --p_shuffle_table")
    if args.gate == "formula" and args.formula_variant in PERCENTILE_FORMULA_VARIANTS and formula_table is None:
        raise SystemExit(f"--formula_variant {args.formula_variant} needs --formula_table")
    if (
        args.q_variant == "senqa"
        or (
            proxy_model is not None
            and str(proxy_model.get("family", "")).startswith("sencache_")
        )
    ) and args.sencache_sensitivity_path is None:
        raise SystemExit("SenCache q/proxy variants need --sencache_sensitivity_path")
    sencache_table_identity = None
    if args.sencache_sensitivity_path is not None:
        table = load_sensitivity_table(args.sencache_sensitivity_path)
        sencache_table_identity = {
            "path": str(args.sencache_sensitivity_path),
            "sha256": table.sha256,
            "metadata": table.metadata,
        }

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
          f"gate={args.gate} p_mode={args.p_mode} q_variant={args.q_variant} "
          f"payload={args.payload_mode} blend={args.payload_blend} "
          f"formula_variant={args.formula_variant} "
          f"thresh={args.threshold} g_max={args.g_max}; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts",
          flush=True)

    teardown = install_gate(pipe, kind=args.gate, threshold=args.threshold,
                            g_max=args.g_max, interval=args.interval,
                            first_enhance=args.first_enhance, num_steps=N,
                            q_abs=q_abs, s_cal=s_cal, a_cal=a_cal, beta=beta,
                            p_mode=str(args.p_mode),
                            p_mix_beta=float(args.p_mix_beta),
                            q_variant=str(args.q_variant),
                            payload_mode=str(args.payload_mode),
                            payload_blend=float(args.payload_blend),
                            formula_variant=str(args.formula_variant),
                            formula_table=formula_table,
                            formula_table_path=(
                                None if args.formula_table is None else str(args.formula_table)
                            ),
                            p_shuffle_table=p_shuffle_table,
                            p_shuffle_seed=int(args.p_shuffle_seed),
                            p_shuffle_scope=str(args.p_shuffle_scope),
                            mem_variant=int(args.mem_variant),
                            target_cached_ratio=float(args.target_cached_ratio),
                            adapt_rate=float(args.adapt_rate),
                            adapt_burn_in=int(args.adapt_burn_in),
                            adapt_overshoot_cap=float(args.adapt_overshoot_cap),
                            proxy_model=proxy_model,
                            proxy_model_path=(None if args.proxy_model is None else str(args.proxy_model)),
                            proxy_mode=str(args.proxy_mode),
                            proxy_cache_threshold=float(args.proxy_cache_threshold),
                            sencache_sensitivity_path=(
                                None if args.sencache_sensitivity_path is None
                                else str(args.sencache_sensitivity_path)
                            ),
                            teacache_backbone=str(args.teacache_backbone),
                            teacache_variant=args.teacache_variant,
                            shadow_depths=tuple(sorted(_parse_int_set(args.shadow_depths))),
                            action_steps=_parse_int_set(args.action_steps))

    per_prompt = []
    try:
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

            reset_gate_state(pipe)
            pipe.transformer.gt_current_prompt_idx = int(global_idx)
            out = pipe(**_pipe_kwargs(prompt, seed, args))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decisions = list(pipe.transformer.gt_decisions)

            out.images[0].save(prompt_dir / "image.png")
            n_cached = sum(1 for d in decisions if d["u"] == 1)
            (prompt_dir / "decisions.json").write_text(
                json.dumps({"prompt_idx": global_idx, "gate": args.gate,
                            "p_mode": str(args.p_mode),
                            "p_mix_beta": float(args.p_mix_beta),
                            "q_variant": str(args.q_variant),
                            "payload_mode": str(args.payload_mode),
                            "payload_blend": float(args.payload_blend),
                            "formula_variant": str(args.formula_variant),
                            "formula_table": (
                                None if args.formula_table is None else str(args.formula_table)
                            ),
                            "proxy_model": (None if args.proxy_model is None
                                            else str(args.proxy_model)),
                            "proxy_mode": str(args.proxy_mode),
                            "proxy_cache_threshold": float(args.proxy_cache_threshold),
                            "sencache_sensitivity_path": (
                                None if args.sencache_sensitivity_path is None
                                else str(args.sencache_sensitivity_path)
                            ),
                            "sencache_sensitivity_sha256": (
                                None if sencache_table_identity is None
                                else sencache_table_identity["sha256"]
                            ),
                            "sencache_sensitivity_metadata": (
                                None if sencache_table_identity is None
                                else sencache_table_identity["metadata"]
                            ),
                            "shadow_depths": sorted(_parse_int_set(args.shadow_depths)),
                            "shadow_extra_compute": bool(args.gate == "shadow_ridge"),
                            "action_steps": sorted(_parse_int_set(args.action_steps)),
                            "p_shuffle_table": (None if args.p_shuffle_table is None
                                                else str(args.p_shuffle_table)),
                            "p_shuffle_seed": int(args.p_shuffle_seed),
                            "p_shuffle_scope": str(args.p_shuffle_scope),
                            "threshold": float(args.threshold), "g_max": args.g_max,
                            "n_cached": n_cached, "n_total": len(decisions),
                            "cached_ratio": n_cached / max(1, len(decisions)),
                            "per_step": decisions}, indent=2, ensure_ascii=False))
            wall = time.perf_counter() - t_p
            manifest_path.write_text(json.dumps({
                "experiment": "state_gate_runner", "gate": args.gate,
                "p_mode": str(args.p_mode),
                "p_mix_beta": float(args.p_mix_beta),
                "q_variant": str(args.q_variant),
                "payload_mode": str(args.payload_mode),
                "payload_blend": float(args.payload_blend),
                "proxy_model": (None if args.proxy_model is None
                                else str(args.proxy_model)),
                "proxy_mode": str(args.proxy_mode),
                "proxy_cache_threshold": float(args.proxy_cache_threshold),
                "sencache_sensitivity_path": (
                    None if args.sencache_sensitivity_path is None
                    else str(args.sencache_sensitivity_path)
                ),
                "sencache_sensitivity_sha256": (
                    None if sencache_table_identity is None
                    else sencache_table_identity["sha256"]
                ),
                "sencache_sensitivity_metadata": (
                    None if sencache_table_identity is None
                    else sencache_table_identity["metadata"]
                ),
                "shadow_depths": sorted(_parse_int_set(args.shadow_depths)),
                "shadow_extra_compute": bool(args.gate == "shadow_ridge"),
                "action_steps": sorted(_parse_int_set(args.action_steps)),
                "p_shuffle_table": (None if args.p_shuffle_table is None
                                    else str(args.p_shuffle_table)),
                "p_shuffle_seed": int(args.p_shuffle_seed),
                "p_shuffle_scope": str(args.p_shuffle_scope),
                "threshold": float(args.threshold), "g_max": int(args.g_max),
                "target_cached_ratio": float(args.target_cached_ratio),
                "adapt_rate": float(args.adapt_rate),
                "prompt_idx": global_idx, "prompt": prompt, "seed": int(seed),
                "num_steps": N, "n_cached": n_cached,
                "cached_ratio": n_cached / max(1, len(decisions)),
                "wall_seconds": wall, "complete": True,
            }, indent=2, ensure_ascii=False))
            per_prompt.append({"global_idx": global_idx,
                               "wall_seconds": wall, "n_cached": n_cached})
            print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
                  f"cached {n_cached}/{len(decisions)} "
                  f"({local_idx + 1}/{len(shard_prompts)})", flush=True)
    finally:
        teardown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "state_gate_runner", "gate": args.gate,
        "p_mode": str(args.p_mode),
        "p_mix_beta": float(args.p_mix_beta),
        "q_variant": str(args.q_variant),
        "payload_mode": str(args.payload_mode),
        "payload_blend": float(args.payload_blend),
        "formula_variant": str(args.formula_variant),
        "formula_table": (None if args.formula_table is None else str(args.formula_table)),
        "proxy_model": (None if args.proxy_model is None
                        else str(args.proxy_model)),
        "proxy_mode": str(args.proxy_mode),
        "proxy_cache_threshold": float(args.proxy_cache_threshold),
        "sencache_sensitivity_path": (
            None if args.sencache_sensitivity_path is None
            else str(args.sencache_sensitivity_path)
        ),
        "sencache_sensitivity_sha256": (
            None if sencache_table_identity is None
            else sencache_table_identity["sha256"]
        ),
        "sencache_sensitivity_metadata": (
            None if sencache_table_identity is None
            else sencache_table_identity["metadata"]
        ),
        "shadow_depths": sorted(_parse_int_set(args.shadow_depths)),
        "shadow_extra_compute": bool(args.gate == "shadow_ridge"),
        "action_steps": sorted(_parse_int_set(args.action_steps)),
        "p_shuffle_table": (None if args.p_shuffle_table is None
                            else str(args.p_shuffle_table)),
        "p_shuffle_seed": int(args.p_shuffle_seed),
        "p_shuffle_scope": str(args.p_shuffle_scope),
        "threshold": float(args.threshold), "g_max": int(args.g_max),
        "num_steps": N, "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count), "base_seed": int(args.seed),
        "model_id": args.model_id,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
