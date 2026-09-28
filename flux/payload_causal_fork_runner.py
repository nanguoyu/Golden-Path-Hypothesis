#!/usr/bin/env python3
"""Same-prefix payload causal fork probe for fixed-schedule SeaCachePayload.

This runner compares two cached-step payloads at the same fork state.  The
prefix is driven by the locked reuse schedule with `payload_mode=reuse`; each
fork then restores the exact same sampler/cache/history state twice and forces
one cached step with either reuse payload or forecast payload.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.a_k_probe import _encode_and_prepare  # noqa: E402
from flux.causal_fork_runner import (  # noqa: E402
    _one_step,
    _parse_ints,
    _restore,
    _snapshot,
)
from flux.trajectory_deviation_runner import (  # noqa: E402
    EPS,
    PAYLOAD_MODES,
    _cos,
    _cache_params_for_args,
    _dot,
    _flowmatch_euler_post,
    _git_commit,
    _load_payload_action_steps,
    _norm,
    _run_full_trace,
    _threshold_for_args,
    install_trajectory_deviation,
    reset_td_state,
)
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402


FIRST_STEP_SCALAR_FIELDS = (
    "action_defect",
    "action_defect_rel_to_full",
    "action_latent_step_defect",
    "action_latent_step_defect_rel",
    "state_gap",
    "state_gap_rel",
    "latent_drift_post",
    "latent_drift_post_rel",
    "output_drift",
    "output_drift_rel",
    "dot_action_gap",
    "cos_action_gap",
    "projection_action_on_total",
)

FIRST_STEP_IMPROVEMENT_FIELDS = (
    "action_defect",
    "action_defect_rel_to_full",
    "action_latent_step_defect",
    "action_latent_step_defect_rel",
    "state_gap",
    "state_gap_rel",
    "latent_drift_post",
    "latent_drift_post_rel",
    "output_drift",
    "output_drift_rel",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--payload_schedule_dir", type=Path, required=True,
                   help="SeaCachePayload(reuse) decisions directory.")
    p.add_argument("--payload_mode", choices=sorted(PAYLOAD_MODES), default="taylor_o1",
                   help="Forecast/control payload for the forecast branch.")
    p.add_argument("--payload_blend", type=float, default=1.0)
    p.add_argument("--payload_sigma", type=float, default=0.5)
    p.add_argument("--payload_control_seed_salt", type=int, default=0,
                   help="Deterministic salt for random/orthogonal payload control directions.")
    p.add_argument("--tail_policies", default="full-after",
                   help="Comma/space separated: full-after, locked-after, or fullH-after (e.g. full2-after).")
    p.add_argument("--fork_steps", default="7,13,20,27,34,39,43,47",
                   help="Candidate cached steps; prompt-specific uncached steps are skipped.")
    p.add_argument("--short_tail_horizons", default="",
                   help="Comma/space separated post-fork horizons m for E_m drift observer. Empty disables.")
    p.add_argument("--state_gap_horizons", default="",
                   help=(
                       "Comma/space separated future horizons m for cumulative state-gap exposure. "
                       "Exposure excludes the forced cache step and sums offsets 1..m."
                   ))
    p.add_argument("--recovery_horizons", default="",
                   help="Comma/space separated post-forced-step update horizons m for recovery rho. Empty disables.")
    p.add_argument("--fork_step_file", type=Path, default=None)
    p.add_argument("--prompt_ids", default=None)
    p.add_argument("--prompt_id_file", type=Path, default=None)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--first_enhance", type=int, default=1)
    p.add_argument("--seacache_thresh", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(mode="SeaCachePayload")
    return p.parse_args()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_selected_ids(args: argparse.Namespace) -> List[int]:
    ids: List[int] = []
    if args.prompt_ids:
        ids.extend(_parse_ints(args.prompt_ids))
    if args.prompt_id_file:
        ids.extend(_parse_ints(args.prompt_id_file.read_text(encoding="utf-8")))
    out: List[int] = []
    seen = set()
    for idx in ids:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(int(idx))
    return sorted(out)


def _run_name_for_args(args: argparse.Namespace, selected_count: int) -> str:
    threshold = str(float(args.seacache_thresh)).replace(".", "")
    blend = str(float(args.payload_blend)).replace(".", "p")
    limit_tag = f"n{args.limit}" if int(args.limit) > 0 else "nfull"
    if selected_count:
        limit_tag += f"_sel{selected_count}"
    salt_tag = ""
    if int(getattr(args, "payload_control_seed_salt", 0)) != 0:
        salt_tag = f"_salt{int(args.payload_control_seed_salt)}"
    return (
        f"payloadfork_seacache_t{threshold}_{args.payload_mode}_b{blend}_"
        f"{limit_tag}{salt_tag}_s{args.seed}_{args.num_steps}"
    )


def _tail_policies(text: str) -> List[str]:
    policies = [tok.strip() for tok in text.replace(",", " ").split() if tok.strip()]
    for policy in policies:
        if policy in {"full-after", "locked-after"}:
            continue
        if policy.startswith("full") and policy.endswith("-after"):
            horizon_text = policy[len("full"):-len("-after")]
            if horizon_text.isdigit():
                continue
        if policy:
            raise ValueError(f"unsupported tail policy: {policy}")
    return policies


def _forced_action_for_tail(tail_policy: str, *, start: int, step_index: int) -> Optional[str]:
    if step_index == start:
        return "cache"
    if tail_policy == "full-after":
        return "full"
    if tail_policy == "locked-after":
        return None
    if tail_policy.startswith("full") and tail_policy.endswith("-after"):
        horizon_text = tail_policy[len("full"):-len("-after")]
        if horizon_text.isdigit():
            horizon = int(horizon_text)
            return "full" if 0 < int(step_index) - int(start) <= horizon else None
    raise ValueError(f"unsupported tail policy: {tail_policy}")


def _native_reuse_with_snapshots(
    pipe,
    ctx: Dict[str, Any],
    *,
    action_steps: Set[int],
    fork_steps: Set[int],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    tr = pipe.transformer
    tr._td_run_kind = "cached"
    tr._td_with_cf = False
    tr._td_payload_mode = "reuse"
    tr._td_payload_blend = 1.0
    reset_td_state(pipe, action_steps=action_steps)
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    snapshots: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i in range(len(ctx["timesteps"])):
            if i in fork_steps and i in action_steps:
                snapshots[i] = _snapshot(pipe, latents, i)
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=None)
            row = {
                "step_index": int(i),
                "decision_u": rec.get("decision_u"),
                "is_cached": bool(rec.get("is_cached")),
                "decision_reason": rec.get("decision_reason"),
                "schedule_locked": rec.get("schedule_locked"),
                "schedule_u": rec.get("schedule_u"),
                "payload_mode": rec.get("payload_mode"),
                "payload_used": rec.get("payload_used"),
                "payload_fallback": rec.get("payload_fallback"),
                "payload_available": rec.get("payload_available"),
            }
            rows.append(row)
            if i in snapshots:
                snapshots[i]["native_action"] = rec.get("decision_u")
                snapshots[i]["native_decision_reason"] = rec.get("decision_reason")
    return latents.detach(), rows, snapshots


def _branch_scalar_fields(
    ctx: Dict[str, Any],
    full_trace: Dict[str, Any],
    *,
    step_index: int,
    z_cached_pre: torch.Tensor,
    z_cached_post: torch.Tensor,
    rec: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the same first-step scalar decomposition used by traj-dev runs."""
    o_drv = rec.get("o_drv")
    if o_drv is None:
        return {key: None for key in FIRST_STEP_SCALAR_FIELDS}

    i = int(step_index)
    sigmas = [float(x) for x in ctx["sigmas"]]
    H_i = sigmas[i + 1] - sigmas[i]
    o_full = full_trace["outputs"][i].to(device=o_drv.device, dtype=torch.float32)
    z_full_post = full_trace["z_post"][i].to(device=z_cached_post.device, dtype=torch.float32)
    o_drv_f = o_drv.detach().to(torch.float32)
    z_cached_pre_f = z_cached_pre.detach().to(torch.float32)
    z_cached_post_f = z_cached_post.detach().to(torch.float32)
    drv_minus_full = o_drv_f - o_full

    out: Dict[str, Any] = {
        "latent_drift_post": _norm(z_cached_post_f - z_full_post),
        "latent_drift_post_rel": _norm(z_cached_post_f - z_full_post) / (_norm(z_full_post) + EPS),
        "output_drift": _norm(drv_minus_full),
        "output_drift_rel": _norm(drv_minus_full) / (_norm(o_full) + EPS),
    }
    o_cf = rec.get("o_cf")
    if o_cf is None:
        for key in (
            "action_defect",
            "action_defect_rel_to_full",
            "action_latent_step_defect",
            "action_latent_step_defect_rel",
            "state_gap",
            "state_gap_rel",
            "dot_action_gap",
            "cos_action_gap",
            "projection_action_on_total",
        ):
            out[key] = None
        return out

    o_cf_f = o_cf.detach().to(torch.float32)
    z_cf_next = _flowmatch_euler_post(z_cached_pre_f, H_i, o_cf_f, o_cf.dtype)
    act = o_drv_f - o_cf_f
    gap = o_cf_f - o_full
    action_latent = z_cached_post_f - z_cf_next
    out.update({
        "action_defect": _norm(act),
        "action_defect_rel_to_full": _norm(act) / (_norm(o_full) + EPS),
        "action_latent_step_defect": _norm(action_latent),
        "action_latent_step_defect_rel": _norm(action_latent) / (_norm(z_cf_next - z_cached_pre_f) + EPS),
        "state_gap": _norm(gap),
        "state_gap_rel": _norm(gap) / (_norm(o_full) + EPS),
        "dot_action_gap": _dot(act, gap),
        "cos_action_gap": _cos(act, gap),
        "projection_action_on_total": _dot(act, drv_minus_full) / (_norm(drv_minus_full) ** 2 + EPS),
    })
    return out


def _run_payload_branch(
    pipe,
    ctx: Dict[str, Any],
    full_trace: Dict[str, Any],
    snap: Dict[str, Any],
    *,
    action_steps: Set[int],
    payload_mode: str,
    payload_blend: float,
    payload_control_seed_salt: int,
    tail_policy: str,
    short_tail_horizons: Sequence[int],
    state_gap_horizons: Sequence[int],
    recovery_horizons: Sequence[int],
) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    tr = pipe.transformer
    tr._td_payload_action_steps = set(int(x) for x in action_steps)
    tr._td_payload_mode = str(payload_mode)
    tr._td_payload_blend = float(payload_blend)
    tr._td_payload_control_seed_salt = int(payload_control_seed_salt)
    old_with_cf = bool(getattr(tr, "_td_with_cf", True))
    tr._td_with_cf = True
    rows: List[Dict[str, Any]] = []
    start = int(snap["step_index"])
    short_tail_drift: Dict[int, float] = {}
    state_gap_exposure: Dict[int, Dict[str, float]] = {
        int(m): {
            "count": 0.0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "sum_rel": 0.0,
            "sum_h": 0.0,
            "max": 0.0,
            "cached_count": 0.0,
            "cached_sum": 0.0,
            "cached_sum_sq": 0.0,
            "cached_sum_rel": 0.0,
            "cached_sum_h": 0.0,
            "cached_max": 0.0,
        }
        for m in state_gap_horizons
    }
    recovery: Dict[int, Dict[str, float]] = {}
    state_gap_pre_vec: Optional[torch.Tensor] = None
    first_update_vec: Optional[torch.Tensor] = None
    first_post_gap_vec: Optional[torch.Tensor] = None
    short_tail_set = {int(x) for x in short_tail_horizons}
    state_gap_set = {int(x) for x in state_gap_horizons}
    recovery_set = {int(x) for x in recovery_horizons}
    try:
        with torch.no_grad():
            for i in range(start, len(ctx["timesteps"])):
                forced = _forced_action_for_tail(tail_policy, start=start, step_index=i)
                latents_pre = latents
                latents, rec = _one_step(pipe, ctx, latents, i, forced_action=forced)
                z_full_pre = full_trace["z_pre"][i].to(device=latents_pre.device, dtype=torch.float32)
                z_full_post = full_trace["z_post"][i].to(device=latents.device, dtype=torch.float32)
                latents_pre_f = latents_pre.detach().to(torch.float32)
                latents_post_f = latents.detach().to(torch.float32)
                branch_update_vec = latents_post_f - latents_pre_f
                full_update_vec = z_full_post - z_full_pre
                update_error_vec = branch_update_vec - full_update_vec
                offset = int(i) - start
                if offset == 0:
                    state_gap_pre_vec = (latents_pre_f - z_full_pre).detach().clone()
                    first_update_vec = branch_update_vec.detach().clone()
                    first_post_gap_vec = (latents_post_f - z_full_post).detach().clone()
                scalar_fields = _branch_scalar_fields(
                    ctx,
                    full_trace,
                    step_index=i,
                    z_cached_pre=latents_pre,
                    z_cached_post=latents,
                    rec=rec,
                )
                rows.append({
                    "step_index": int(i),
                    "forced_action": forced,
                    "decision_u": rec.get("decision_u"),
                    "is_cached": bool(rec.get("is_cached")),
                    "decision_reason": rec.get("decision_reason"),
                    "schedule_locked": rec.get("schedule_locked"),
                    "schedule_u": rec.get("schedule_u"),
                    "native_u": rec.get("native_u"),
                    "payload_mode": rec.get("payload_mode"),
                    "payload_base_mode": rec.get("payload_base_mode"),
                    "payload_control": rec.get("payload_control"),
                    "payload_used": rec.get("payload_used"),
                    "payload_available": rec.get("payload_available"),
                    "payload_fallback": rec.get("payload_fallback"),
                    "payload_delta_from_reuse_norm": rec.get("payload_delta_from_reuse_norm"),
                    **scalar_fields,
                })
                short_tail_horizon = int(offset) + 1
                if short_tail_horizon in short_tail_set:
                    drift = _maybe_float(scalar_fields.get("latent_drift_post"))
                    if drift is not None:
                        short_tail_drift[short_tail_horizon] = float(drift)
                if offset > 0 and state_gap_set:
                    state_gap = _maybe_float(scalar_fields.get("state_gap"))
                    state_gap_rel = _maybe_float(scalar_fields.get("state_gap_rel"))
                    if state_gap is not None:
                        sigmas = [float(x) for x in ctx["sigmas"]]
                        h_abs = abs(float(sigmas[i + 1] - sigmas[i]))
                        is_cached_step = bool(rec.get("is_cached"))
                        for horizon in state_gap_set:
                            if offset <= int(horizon):
                                rec_exp = state_gap_exposure[int(horizon)]
                                rec_exp["count"] += 1.0
                                rec_exp["sum"] += float(state_gap)
                                rec_exp["sum_sq"] += float(state_gap) * float(state_gap)
                                rec_exp["sum_h"] += float(h_abs) * float(state_gap)
                                rec_exp["max"] = max(float(rec_exp["max"]), float(state_gap))
                                if state_gap_rel is not None:
                                    rec_exp["sum_rel"] += float(state_gap_rel)
                                if is_cached_step:
                                    rec_exp["cached_count"] += 1.0
                                    rec_exp["cached_sum"] += float(state_gap)
                                    rec_exp["cached_sum_sq"] += float(state_gap) * float(state_gap)
                                    rec_exp["cached_sum_h"] += float(h_abs) * float(state_gap)
                                    rec_exp["cached_max"] = max(float(rec_exp["cached_max"]), float(state_gap))
                                    if state_gap_rel is not None:
                                        rec_exp["cached_sum_rel"] += float(state_gap_rel)
                if offset > 0 and offset in recovery_set and first_post_gap_vec is not None:
                    den = _norm(first_post_gap_vec) ** 2 + EPS
                    dot = _dot(update_error_vec, first_post_gap_vec)
                    recovery[int(offset)] = {
                        "rho": float(-dot / den),
                        "dot": float(dot),
                        "den": float(den),
                        "update_error_norm": float(_norm(update_error_vec)),
                        "first_gap_norm": float(_norm(first_post_gap_vec)),
                    }
    finally:
        tr._td_with_cf = old_with_cf
    return {
        "final": latents.detach(),
        "rows": rows,
        "short_tail_drift": short_tail_drift,
        "state_gap_exposure": state_gap_exposure,
        "recovery": recovery,
        "state_gap_pre_vec": state_gap_pre_vec,
        "first_update_vec": first_update_vec,
    }


def _hamming(a: List[str], b: List[str]) -> int:
    return int(sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)))


def _maybe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _positive_horizons(values: Sequence[int], *, num_steps: int, name: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        horizon = int(value)
        if horizon <= 0:
            raise ValueError(f"{name} horizons must be positive; got {horizon}")
        if horizon >= int(num_steps):
            raise ValueError(f"{name} horizon {horizon} is outside num_steps={num_steps}")
        if horizon in seen:
            continue
        seen.add(horizon)
        out.append(horizon)
    return sorted(out)


def _sign(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    if float(value) > 0.0:
        return 1
    if float(value) < 0.0:
        return -1
    return 0


def _anti_reinforcement_fields(
    reuse: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    s_vec = reuse.get("state_gap_pre_vec")
    reuse_update = reuse.get("first_update_vec")
    forecast_update = forecast.get("first_update_vec")
    if s_vec is None or reuse_update is None or forecast_update is None:
        return {
            "state_gap_pre_norm": None,
            "delta_update_forecast_minus_reuse_norm": None,
            "dot_delta_update_state_gap_pre": None,
            "cos_delta_update_state_gap_pre": None,
            "anti_reinforcement_energy": None,
            "anti_reinforcement_energy_rel": None,
            "anti_reinforcement_predicts_improvement": None,
        }
    s = s_vec.detach().to(torch.float32)
    delta = forecast_update.detach().to(torch.float32) - reuse_update.detach().to(torch.float32)
    s_norm = _norm(s)
    delta_norm = _norm(delta)
    dot = _dot(delta, s)
    energy = 2.0 * dot + delta_norm * delta_norm
    return {
        "state_gap_pre_norm": s_norm,
        "delta_update_forecast_minus_reuse_norm": delta_norm,
        "dot_delta_update_state_gap_pre": dot,
        "cos_delta_update_state_gap_pre": _cos(delta, s),
        "anti_reinforcement_energy": energy,
        "anti_reinforcement_energy_rel": energy / (s_norm * s_norm + EPS),
        "anti_reinforcement_predicts_improvement": bool(energy < 0.0),
    }


def _short_tail_fields(
    reuse: Dict[str, Any],
    forecast: Dict[str, Any],
    *,
    horizons: Sequence[int],
    terminal_improvement: Optional[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    terminal_sign = _sign(terminal_improvement)
    reuse_tail = reuse.get("short_tail_drift", {})
    forecast_tail = forecast.get("short_tail_drift", {})
    for horizon in horizons:
        m = int(horizon)
        reuse_value = _maybe_float(reuse_tail.get(m))
        forecast_value = _maybe_float(forecast_tail.get(m))
        improvement = (
            None if reuse_value is None or forecast_value is None
            else float(reuse_value - forecast_value)
        )
        improvement_sign = _sign(improvement)
        out[f"reuse_short_tail_available_m{m}"] = reuse_value is not None
        out[f"forecast_short_tail_available_m{m}"] = forecast_value is not None
        out[f"reuse_short_tail_drift_m{m}"] = reuse_value
        out[f"forecast_short_tail_drift_m{m}"] = forecast_value
        out[f"short_tail_improvement_m{m}"] = improvement
        out[f"short_tail_matches_terminal_m{m}"] = (
            None
            if terminal_sign is None or improvement_sign is None or terminal_sign == 0 or improvement_sign == 0
            else bool(terminal_sign == improvement_sign)
        )
    return out


def _state_gap_exposure_fields(
    reuse: Dict[str, Any],
    forecast: Dict[str, Any],
    *,
    horizons: Sequence[int],
    terminal_improvement: Optional[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    terminal_sign = _sign(terminal_improvement)
    reuse_exp = reuse.get("state_gap_exposure", {})
    forecast_exp = forecast.get("state_gap_exposure", {})
    for horizon in horizons:
        m = int(horizon)
        r = reuse_exp.get(m) if isinstance(reuse_exp, dict) else None
        f = forecast_exp.get(m) if isinstance(forecast_exp, dict) else None
        r = r if isinstance(r, dict) else {}
        f = f if isinstance(f, dict) else {}
        reuse_count = _maybe_float(r.get("count"))
        forecast_count = _maybe_float(f.get("count"))
        reuse_sum = _maybe_float(r.get("sum"))
        forecast_sum = _maybe_float(f.get("sum"))
        reuse_sum_sq = _maybe_float(r.get("sum_sq"))
        forecast_sum_sq = _maybe_float(f.get("sum_sq"))
        reuse_sum_rel = _maybe_float(r.get("sum_rel"))
        forecast_sum_rel = _maybe_float(f.get("sum_rel"))
        reuse_sum_h = _maybe_float(r.get("sum_h"))
        forecast_sum_h = _maybe_float(f.get("sum_h"))
        reuse_max = _maybe_float(r.get("max"))
        forecast_max = _maybe_float(f.get("max"))
        reuse_cached_count = _maybe_float(r.get("cached_count"))
        forecast_cached_count = _maybe_float(f.get("cached_count"))
        reuse_cached_sum = _maybe_float(r.get("cached_sum"))
        forecast_cached_sum = _maybe_float(f.get("cached_sum"))
        reuse_cached_sum_sq = _maybe_float(r.get("cached_sum_sq"))
        forecast_cached_sum_sq = _maybe_float(f.get("cached_sum_sq"))
        reuse_cached_sum_rel = _maybe_float(r.get("cached_sum_rel"))
        forecast_cached_sum_rel = _maybe_float(f.get("cached_sum_rel"))
        reuse_cached_sum_h = _maybe_float(r.get("cached_sum_h"))
        forecast_cached_sum_h = _maybe_float(f.get("cached_sum_h"))
        reuse_cached_max = _maybe_float(r.get("cached_max"))
        forecast_cached_max = _maybe_float(f.get("cached_max"))
        improvement = (
            None if reuse_sum is None or forecast_sum is None
            else float(reuse_sum - forecast_sum)
        )
        improvement_sq = (
            None if reuse_sum_sq is None or forecast_sum_sq is None
            else float(reuse_sum_sq - forecast_sum_sq)
        )
        improvement_rel = (
            None if reuse_sum_rel is None or forecast_sum_rel is None
            else float(reuse_sum_rel - forecast_sum_rel)
        )
        improvement_h = (
            None if reuse_sum_h is None or forecast_sum_h is None
            else float(reuse_sum_h - forecast_sum_h)
        )
        cached_improvement = (
            None if reuse_cached_sum is None or forecast_cached_sum is None
            else float(reuse_cached_sum - forecast_cached_sum)
        )
        cached_improvement_sq = (
            None if reuse_cached_sum_sq is None or forecast_cached_sum_sq is None
            else float(reuse_cached_sum_sq - forecast_cached_sum_sq)
        )
        cached_improvement_rel = (
            None if reuse_cached_sum_rel is None or forecast_cached_sum_rel is None
            else float(reuse_cached_sum_rel - forecast_cached_sum_rel)
        )
        cached_improvement_h = (
            None if reuse_cached_sum_h is None or forecast_cached_sum_h is None
            else float(reuse_cached_sum_h - forecast_cached_sum_h)
        )
        improvement_sign = _sign(improvement)
        cached_improvement_sign = _sign(cached_improvement)
        out[f"reuse_state_gap_exposure_available_m{m}"] = bool(reuse_count and reuse_count > 0)
        out[f"forecast_state_gap_exposure_available_m{m}"] = bool(forecast_count and forecast_count > 0)
        out[f"reuse_state_gap_exposure_count_m{m}"] = reuse_count
        out[f"forecast_state_gap_exposure_count_m{m}"] = forecast_count
        out[f"reuse_state_gap_exposure_m{m}"] = reuse_sum
        out[f"forecast_state_gap_exposure_m{m}"] = forecast_sum
        out[f"state_gap_exposure_improvement_m{m}"] = improvement
        out[f"reuse_state_gap_exposure_sq_m{m}"] = reuse_sum_sq
        out[f"forecast_state_gap_exposure_sq_m{m}"] = forecast_sum_sq
        out[f"state_gap_exposure_sq_improvement_m{m}"] = improvement_sq
        out[f"reuse_state_gap_rel_exposure_m{m}"] = reuse_sum_rel
        out[f"forecast_state_gap_rel_exposure_m{m}"] = forecast_sum_rel
        out[f"state_gap_rel_exposure_improvement_m{m}"] = improvement_rel
        out[f"reuse_state_gap_h_exposure_m{m}"] = reuse_sum_h
        out[f"forecast_state_gap_h_exposure_m{m}"] = forecast_sum_h
        out[f"state_gap_h_exposure_improvement_m{m}"] = improvement_h
        out[f"reuse_state_gap_exposure_max_m{m}"] = reuse_max
        out[f"forecast_state_gap_exposure_max_m{m}"] = forecast_max
        out[f"state_gap_exposure_matches_terminal_m{m}"] = (
            None
            if terminal_sign is None or improvement_sign is None or terminal_sign == 0 or improvement_sign == 0
            else bool(terminal_sign == improvement_sign)
        )
        out[f"reuse_cached_state_gap_exposure_available_m{m}"] = bool(
            reuse_cached_count and reuse_cached_count > 0
        )
        out[f"forecast_cached_state_gap_exposure_available_m{m}"] = bool(
            forecast_cached_count and forecast_cached_count > 0
        )
        out[f"reuse_cached_state_gap_exposure_count_m{m}"] = reuse_cached_count
        out[f"forecast_cached_state_gap_exposure_count_m{m}"] = forecast_cached_count
        out[f"reuse_cached_state_gap_exposure_m{m}"] = reuse_cached_sum
        out[f"forecast_cached_state_gap_exposure_m{m}"] = forecast_cached_sum
        out[f"cached_state_gap_exposure_improvement_m{m}"] = cached_improvement
        out[f"reuse_cached_state_gap_exposure_sq_m{m}"] = reuse_cached_sum_sq
        out[f"forecast_cached_state_gap_exposure_sq_m{m}"] = forecast_cached_sum_sq
        out[f"cached_state_gap_exposure_sq_improvement_m{m}"] = cached_improvement_sq
        out[f"reuse_cached_state_gap_rel_exposure_m{m}"] = reuse_cached_sum_rel
        out[f"forecast_cached_state_gap_rel_exposure_m{m}"] = forecast_cached_sum_rel
        out[f"cached_state_gap_rel_exposure_improvement_m{m}"] = cached_improvement_rel
        out[f"reuse_cached_state_gap_h_exposure_m{m}"] = reuse_cached_sum_h
        out[f"forecast_cached_state_gap_h_exposure_m{m}"] = forecast_cached_sum_h
        out[f"cached_state_gap_h_exposure_improvement_m{m}"] = cached_improvement_h
        out[f"reuse_cached_state_gap_exposure_max_m{m}"] = reuse_cached_max
        out[f"forecast_cached_state_gap_exposure_max_m{m}"] = forecast_cached_max
        out[f"cached_state_gap_exposure_matches_terminal_m{m}"] = (
            None
            if (
                terminal_sign is None
                or cached_improvement_sign is None
                or terminal_sign == 0
                or cached_improvement_sign == 0
            )
            else bool(terminal_sign == cached_improvement_sign)
        )
    return out


def _recovery_fields(
    reuse: Dict[str, Any],
    forecast: Dict[str, Any],
    *,
    horizons: Sequence[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    branches = (("reuse", reuse), ("forecast", forecast))
    for horizon in horizons:
        m = int(horizon)
        branch_rhos: Dict[str, Optional[float]] = {}
        for branch_name, branch in branches:
            rec = branch.get("recovery", {}).get(m)
            available = isinstance(rec, dict)
            out[f"{branch_name}_recovery_available_m{m}"] = available
            for key in ("rho", "dot", "den", "update_error_norm", "first_gap_norm"):
                value = None if not available else _maybe_float(rec.get(key))
                out[f"{branch_name}_recovery_{key}_m{m}"] = value
                if key == "rho":
                    branch_rhos[branch_name] = value
        reuse_rho = branch_rhos.get("reuse")
        forecast_rho = branch_rhos.get("forecast")
        out[f"recovery_delta_rho_m{m}"] = (
            None if reuse_rho is None or forecast_rho is None
            else float(forecast_rho - reuse_rho)
        )
    return out


def _first_step_scalar_summary(
    reuse_first: Dict[str, Any],
    forecast_first: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in FIRST_STEP_SCALAR_FIELDS:
        reuse_value = _maybe_float(reuse_first.get(key))
        forecast_value = _maybe_float(forecast_first.get(key))
        out[f"reuse_first_{key}"] = reuse_value
        out[f"forecast_first_{key}"] = forecast_value
        out[f"first_delta_{key}"] = (
            None if reuse_value is None or forecast_value is None
            else float(forecast_value - reuse_value)
        )
    for key in FIRST_STEP_IMPROVEMENT_FIELDS:
        reuse_value = _maybe_float(reuse_first.get(key))
        forecast_value = _maybe_float(forecast_first.get(key))
        out[f"first_improvement_{key}"] = (
            None if reuse_value is None or forecast_value is None
            else float(reuse_value - forecast_value)
        )
    return out


def _paired_rows(
    pipe,
    ctx: Dict[str, Any],
    *,
    full_trace: Dict[str, Any],
    full_final: torch.Tensor,
    native_final: torch.Tensor,
    native_rows: List[Dict[str, Any]],
    snapshots: Dict[int, Dict[str, Any]],
    action_steps: Set[int],
    prompt_id: int,
    seed: int,
    payload_mode: str,
    payload_blend: float,
    payload_control_seed_salt: int,
    tail_policies: List[str],
    short_tail_horizons: Sequence[int],
    state_gap_horizons: Sequence[int],
    recovery_horizons: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    z_full = full_final.to(torch.float32)
    z_native = native_final.to(torch.float32)
    native_seq = ["C" if row.get("is_cached") else "F" for row in native_rows]
    for fork_step, snap in sorted(snapshots.items()):
        for tail_policy in tail_policies:
            base = {
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "fork_step": int(fork_step),
                "tail_policy": tail_policy,
                "forecast_payload_mode": str(payload_mode),
                "forecast_payload_blend": float(payload_blend),
                "forecast_payload_control_seed_salt": int(payload_control_seed_salt),
                "native_action": snap.get("native_action"),
                "native_decision_reason": snap.get("native_decision_reason"),
                "snapshot_cache_digest": snap.get("cache_state_digest"),
                "snapshot_scheduler_digest": snap.get("scheduler_state_digest"),
                "snapshot_rng_digest": snap.get("rng_state_digest"),
            }
            try:
                reuse = _run_payload_branch(
                    pipe, ctx, full_trace, snap,
                    action_steps=action_steps,
                    payload_mode="reuse",
                    payload_blend=1.0,
                    payload_control_seed_salt=0,
                    tail_policy=tail_policy,
                    short_tail_horizons=short_tail_horizons,
                    state_gap_horizons=state_gap_horizons,
                    recovery_horizons=recovery_horizons,
                )
                forecast = _run_payload_branch(
                    pipe, ctx, full_trace, snap,
                    action_steps=action_steps,
                    payload_mode=payload_mode,
                    payload_blend=payload_blend,
                    payload_control_seed_salt=int(payload_control_seed_salt),
                    tail_policy=tail_policy,
                    short_tail_horizons=short_tail_horizons,
                    state_gap_horizons=state_gap_horizons,
                    recovery_horizons=recovery_horizons,
                )
            except RuntimeError as exc:
                out.append({**base, "valid": False, "error": repr(exc)})
                continue

            z_reuse = reuse["final"].to(torch.float32)
            z_forecast = forecast["final"].to(torch.float32)
            reuse_seq = ["C" if row.get("is_cached") else "F" for row in reuse["rows"]]
            forecast_seq = ["C" if row.get("is_cached") else "F" for row in forecast["rows"]]
            native_suffix = native_seq[fork_step:]
            reuse_first = reuse["rows"][0] if reuse["rows"] else {}
            forecast_first = forecast["rows"][0] if forecast["rows"] else {}
            reuse_drift = _norm(z_reuse - z_full)
            forecast_drift = _norm(z_forecast - z_full)
            improvement = reuse_drift - forecast_drift
            out.append({
                **base,
                "valid": True,
                "error": "",
                "reuse_final_drift": reuse_drift,
                "forecast_final_drift": forecast_drift,
                "improvement_reuse_minus_forecast": improvement,
                "forecast_reuse_final_l2": _norm(z_forecast - z_reuse),
                "native_final_drift": _norm(z_native - z_full),
                "reuse_native_final_l2": _norm(z_reuse - z_native),
                "forecast_native_final_l2": _norm(z_forecast - z_native),
                "reuse_first_is_cached": bool(reuse_first.get("is_cached")),
                "forecast_first_is_cached": bool(forecast_first.get("is_cached")),
                "reuse_first_decision_u": reuse_first.get("decision_u"),
                "forecast_first_decision_u": forecast_first.get("decision_u"),
                "reuse_first_payload_used": reuse["rows"][0].get("payload_used") if reuse["rows"] else None,
                "forecast_first_payload_used": forecast["rows"][0].get("payload_used") if forecast["rows"] else None,
                "forecast_first_payload_available": forecast["rows"][0].get("payload_available") if forecast["rows"] else None,
                "forecast_first_payload_fallback": forecast["rows"][0].get("payload_fallback") if forecast["rows"] else None,
                **_first_step_scalar_summary(reuse_first, forecast_first),
                **_anti_reinforcement_fields(reuse, forecast),
                **_short_tail_fields(
                    reuse,
                    forecast,
                    horizons=short_tail_horizons,
                    terminal_improvement=improvement,
                ),
                **_state_gap_exposure_fields(
                    reuse,
                    forecast,
                    horizons=state_gap_horizons,
                    terminal_improvement=improvement,
                ),
                **_recovery_fields(reuse, forecast, horizons=recovery_horizons),
                "future_action_hamming_forecast_vs_reuse": _hamming(forecast_seq, reuse_seq),
                "reuse_action_hamming_vs_native_suffix": _hamming(reuse_seq, native_suffix),
                "forecast_action_hamming_vs_native_suffix": _hamming(forecast_seq, native_suffix),
                "reuse_cache_rate_suffix": float(sum(x == "C" for x in reuse_seq) / max(len(reuse_seq), 1)),
                "forecast_cache_rate_suffix": float(sum(x == "C" for x in forecast_seq) / max(len(forecast_seq), 1)),
            })
    return out


def _run_prompt(pipe, prompt: str, prompt_id: int, seed: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    action_steps = _load_payload_action_steps(args.payload_schedule_dir, prompt_id)
    fork_steps = set(int(x) for x in args.fork_steps_resolved)
    full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
    native_final, native_rows, snapshots = _native_reuse_with_snapshots(
        pipe,
        ctx,
        action_steps=action_steps,
        fork_steps=fork_steps,
    )
    rows = _paired_rows(
        pipe,
        ctx,
        full_trace=full_trace,
        full_final=full_trace["final"],
        native_final=native_final,
        native_rows=native_rows,
        snapshots=snapshots,
        action_steps=action_steps,
        prompt_id=prompt_id,
        seed=seed,
        payload_mode=args.payload_mode,
        payload_blend=float(args.payload_blend),
        payload_control_seed_salt=int(args.payload_control_seed_salt),
        tail_policies=args.tail_policies_resolved,
        short_tail_horizons=args.short_tail_horizons_resolved,
        state_gap_horizons=args.state_gap_horizons_resolved,
        recovery_horizons=args.recovery_horizons_resolved,
    )
    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prompt_dir / "native_reuse_rows.jsonl", native_rows)
    _write_csv(prompt_dir / "payload_causal_fork_rows.csv", rows)
    (prompt_dir / "manifest.json").write_text(
        json.dumps({
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "num_steps": int(args.num_steps),
            "payload_schedule_dir": str(args.payload_schedule_dir),
            "forecast_payload_mode": str(args.payload_mode),
            "forecast_payload_blend": float(args.payload_blend),
            "forecast_payload_control_seed_salt": int(args.payload_control_seed_salt),
            "fork_steps_requested": [int(x) for x in args.fork_steps_resolved],
            "fork_steps_materialized": sorted(int(x) for x in snapshots),
            "tail_policies": args.tail_policies_resolved,
            "short_tail_horizons": [int(x) for x in args.short_tail_horizons_resolved],
            "state_gap_horizons": [int(x) for x in args.state_gap_horizons_resolved],
            "recovery_horizons": [int(x) for x in args.recovery_horizons_resolved],
            "complete": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    args = parse_args()
    if args.payload_mode == "reuse" and float(args.payload_blend) >= 1.0:
        raise ValueError("forecast branch payload must differ from reuse")
    args.fork_steps_resolved = [
        step for step in _parse_ints(args.fork_steps, args.fork_step_file)
        if 0 <= int(step) < int(args.num_steps)
    ]
    if not args.fork_steps_resolved:
        raise ValueError("no valid fork steps")
    args.tail_policies_resolved = _tail_policies(args.tail_policies)
    args.short_tail_horizons_resolved = _positive_horizons(
        _parse_ints(args.short_tail_horizons),
        num_steps=int(args.num_steps),
        name="short_tail",
    )
    args.state_gap_horizons_resolved = _positive_horizons(
        _parse_ints(args.state_gap_horizons),
        num_steps=int(args.num_steps),
        name="state_gap",
    )
    args.recovery_horizons_resolved = _positive_horizons(
        _parse_ints(args.recovery_horizons),
        num_steps=int(args.num_steps),
        name="recovery",
    )
    selected_ids = _parse_selected_ids(args)
    if args.run_name is None:
        args.run_name = _run_name_for_args(args, len(selected_ids))

    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    if selected_ids:
        max_id = len(prompts_all) - 1
        missing = [idx for idx in selected_ids if idx > max_id]
        if missing:
            raise ValueError(f"prompt IDs outside loaded prompt range 0..{max_id}: {missing[:10]}")
        prompt_records = [(idx, prompts_all[idx]) for idx in selected_ids]
    else:
        prompt_records = list(enumerate(prompts_all))
    start, end = split_shard(len(prompt_records), args.shard_count, args.shard_idx)
    shard_records = prompt_records[start:end]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not shard_records:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.")
        return 0

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    teardown = install_trajectory_deviation(
        pipe,
        mode="SeaCachePayload",
        threshold=_threshold_for_args(args),
        num_steps=int(args.num_steps),
        first_enhance=int(args.first_enhance),
        payload_mode="reuse",
        payload_blend=1.0,
        payload_sigma=float(args.payload_sigma),
    )

    all_rows: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []
    try:
        for prompt_id, prompt in shard_records:
            prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
            if args.resume and (prompt_dir / "manifest.json").is_file():
                print(f"[shard {args.shard_idx}] prompt {prompt_id} complete, skip", flush=True)
                continue
            t_prompt = time.perf_counter()
            seed = seed_for(args.seed, prompt_id)
            rows = _run_prompt(pipe, prompt, prompt_id, seed, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_prompt
            all_rows.extend(rows)
            per_image_records.append({"idx": int(prompt_id), "denoise_s": float(dt), "decode_s": 0.0})
            valid = sum(1 for row in rows if row.get("valid") is True)
            print(f"[shard {args.shard_idx}] prompt {prompt_id} done in {dt:.1f}s valid={valid}/{len(rows)}",
                  flush=True)
    finally:
        teardown()

    _write_csv(args.output_dir / f"payload_causal_fork_rows_shard{args.shard_idx}of{args.shard_count}.csv", all_rows)
    git_sha, git_dirty = _git_commit()
    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "mode": "SeaCachePayload",
        "cache_params": {
            "threshold": _threshold_for_args(args),
            **_cache_params_for_args(args),
            "payload_schedule_dir": str(args.payload_schedule_dir),
        },
        "forecast_payload_mode": str(args.payload_mode),
        "forecast_payload_blend": float(args.payload_blend),
        "forecast_payload_control_seed_salt": int(args.payload_control_seed_salt),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "prompt_file": str(args.prompt_file),
        "limit": int(args.limit),
        "selected_prompt_ids": [int(x) for x in selected_ids],
        "fork_steps": [int(x) for x in args.fork_steps_resolved],
        "tail_policies": args.tail_policies_resolved,
        "short_tail_horizons": [int(x) for x in args.short_tail_horizons_resolved],
        "state_gap_horizons": [int(x) for x in args.state_gap_horizons_resolved],
        "recovery_horizons": [int(x) for x in args.recovery_horizons_resolved],
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
    }
    (args.output_dir / f"manifest_shard{args.shard_idx}of{args.shard_count}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_timing_json(
        args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json",
        per_image=per_image_records,
        config={
            "cache_mode": "payload_causal_fork_SeaCachePayload",
            "num_steps": int(args.num_steps),
            "seed": int(args.seed),
            "shard_idx": int(args.shard_idx),
            "shard_count": int(args.shard_count),
        },
        model_load_s=model_load_end - t0,
        wallclock_total_s=time.perf_counter() - t0,
        device=device,
    )
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
