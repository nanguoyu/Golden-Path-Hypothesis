#!/usr/bin/env python3
"""E0 collector for online proxy features and FA/SL/RE fork labels.

This runner is deliberately outside the locked cache baselines.  It reuses the
trajectory-deviation monkeypatch to expose a strict closed-state fork surface:

* prefix state is the native cached rollout state ``X_k``;
* ``v_cf`` is the forced-full velocity on that same cached state;
* forced candidate action defect is ``v_cache_candidate - v_cf``;
* labels ``Y_FA``, ``Y_SL`` and ``Y_RE`` are collected separately.

The row schema keeps online-visible decision-time features separate from
oracle labels and diagnostics.  The audit script enforces that separation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.a_k_probe import _encode_and_prepare  # noqa: E402
from flux.causal_fork_runner import (  # noqa: E402
    _hamming,
    _native_with_snapshots,
    _one_step,
    _parse_ints,
    _restore,
)
from flux.trajectory_deviation_runner import (  # noqa: E402
    EPS,
    _cache_params_for_args,
    _git_commit,
    _norm,
    _parse_int_list,
    _parse_prompt_id_tokens,
    _run_full_trace,
    _threshold_for_args,
    install_trajectory_deviation,
)
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402
from lib.sencache import load_sensitivity_table  # noqa: E402


BASE_ONLINE_FEATURE_ALLOWLIST = [
    "mode",
    "cache_threshold",
    "fork_step",
    "step_index",
    "online_last_refresh_step_pre",
    "online_cache_age_pre",
    "online_cache_run_len_pre",
    "online_num_cached_so_far_pre",
    "online_cache_ratio_so_far_pre",
    "online_c_stale_pre",
    "online_c_traj_pre",
    "online_c_mem_pre",
    "online_step_size_abs",
    "online_gate_increment_raw",
    "online_gate_increment_rescaled",
    "online_gate_accumulator_before",
    "online_threshold_margin_before",
    "online_threshold_margin_after",
    "online_p_path_decision",
    "online_p_disp_decision",
    "online_p_tortuosity_log",
    "online_q_proxy_no_sa",
    "online_previous_modulated_input_norm",
    "online_previous_residual_norm",
    "online_gate_diff_prev_residual_cos",
    "online_gate_diff_present",
    "online_gate_diff_norm",
    "online_gate_diff_sketch_l2",
    "online_prev_residual_present",
    "online_prev_residual_norm",
    "online_prev_residual_sketch_l2",
    "online_sencache_anchor_present_pre",
    "online_sencache_anchor_step_pre",
    "online_sencache_anchor_timestep_pre",
    "online_sencache_table_index_pre",
    "online_sencache_table_timestep_pre",
    "online_sencache_table_abs_timestep_error_pre",
    "online_sencache_j_x_norm_pre",
    "online_sencache_j_t_norm_pre",
    "online_sencache_delta_latent_norm_pre",
    "online_sencache_delta_t_abs_pre",
    "online_sencache_latent_term_pre",
    "online_sencache_timestep_term_pre",
    "online_sencache_score_pre",
    "online_sencache_score_log1p_pre",
    "online_fd_anchor_present_pre",
    "online_fd_anchor_step_pre",
    "online_fd_anchor_gap_pre",
    "online_fd_order_avail_pre",
    "online_fd_d0_norm_pre",
    "online_fd_d1_norm_pre",
    "online_fd_d1_rel_pre",
    "online_fd_d2_norm_pre",
    "online_fd_d2_rel_pre",
    "online_fd_d1_d2_cos_pre",
    "online_fd_taylor_o1_drift_norm_pre",
    "online_fd_taylor_o1_drift_rel_pre",
    "online_fd_taylor_o2_drift_norm_pre",
    "online_fd_taylor_o2_drift_rel_pre",
    "online_fd_taylor_o2_minus_o1_norm_pre",
    "online_fd_taylor_o2_minus_o1_rel_pre",
    "online_fd_hicache_o2_minus_taylor_o2_norm_pre",
    "online_fd_hicache_o2_minus_taylor_o2_rel_pre",
    "online_fd_hicache_o2_minus_o1_norm_pre",
    "online_fd_hicache_o2_minus_o1_rel_pre",
    "online_fd_taylor_o1_innovation_ema_pre",
    "online_fd_taylor_o2_innovation_ema_pre",
    "online_fd_hicache_o2_innovation_ema_pre",
    "online_fd_innovation_update_count_pre",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX online proxy E0 collector.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--mode", choices=["SeaCache", "TeaCache", "SenCache"], default="SeaCache")
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
    p.add_argument("--teacache_thresh", type=float, default=0.3)
    p.add_argument("--teacache_backbone", default="flux")
    p.add_argument("--teacache_variant", default=None)
    p.add_argument("--sencache_sensitivity_path", type=Path, default=None)
    p.add_argument("--sencache_thresh", type=float, default=None,
                   help="Alias for --sencache_thresh_main.")
    p.add_argument("--sencache_thresh_start", type=float, default=0.005)
    p.add_argument("--sencache_thresh_main", type=float, default=None)
    p.add_argument("--sencache_K", type=int, default=10)
    p.add_argument("--sencache_threshold_scale", default="auto")
    p.add_argument("--sencache_switch_ratio", type=float, default=0.2)
    p.add_argument("--sencache_ret_steps", type=int, default=0)
    p.add_argument("--sencache_cutoff_steps", type=int, default=-1)
    p.add_argument("--fork_steps", default="6,15,18,25,35,45")
    p.add_argument("--fork_step_file", type=Path, default=None)
    p.add_argument("--dense_all_steps", action="store_true",
                   help=("Collect fork labels for every cache-eligible step. "
                         "For FLUX N=50 and first_enhance=1 this is 1..48."))
    p.add_argument("--prompt_ids", default=None)
    p.add_argument("--prompt_id_file", type=Path, default=None)
    p.add_argument("--online_direction_sketch_dims", type=int, default=0)
    p.add_argument("--shadow_depths", default="")
    p.add_argument("--shadow_on", choices=["all_steps", "cache_candidates"], default="all_steps")
    p.add_argument("--non_pollution_check", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--non_pollution", dest="non_pollution_check", action="store_true")
    p.add_argument("--no_non_pollution", "--no-non_pollution", dest="non_pollution_check", action="store_false")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _eligible_fork_steps(num_steps: int, first_enhance: int) -> List[int]:
    start = max(1, int(first_enhance))
    stop = max(start, int(num_steps) - 1)
    return list(range(start, stop))


def _resolve_fork_steps(args: argparse.Namespace) -> Tuple[List[int], str]:
    dense_tokens = {"all", "all_eligible", "dense", "dense_all_steps"}
    if args.dense_all_steps or str(args.fork_steps).strip().lower() in dense_tokens:
        return _eligible_fork_steps(args.num_steps, args.first_enhance), "dense_all_steps"
    steps = [
        s for s in _parse_ints(args.fork_steps, args.fork_step_file)
        if 0 <= s < int(args.num_steps)
    ]
    return steps, "explicit"


def _run_name_for_args(args: argparse.Namespace, selected_prompt_count: int) -> str:
    if args.mode == "TeaCache":
        method = "teacache"
    elif args.mode == "SenCache":
        method = "sencache"
    else:
        method = "seacache"
    t_tag = str(_threshold_for_args(args)).replace(".", "")
    limit_tag = f"n{args.limit}" if args.limit > 0 else "nfull"
    if selected_prompt_count:
        limit_tag += f"_sel{selected_prompt_count}"
    return f"onlineproxy_e0_{method}_t{t_tag}_{limit_tag}_s{args.seed}_{args.num_steps}"


def _parse_prompt_ids(text: Optional[str], path: Optional[Path]) -> List[int]:
    ids: List[int] = []
    if text:
        ids.extend(_parse_prompt_id_tokens(text))
    if path:
        ids.extend(_parse_prompt_id_tokens(path.read_text(encoding="utf-8")))
    out: List[int] = []
    seen = set()
    for idx in ids:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(int(idx))
    return sorted(out)


def _online_feature_allowlist(row_keys: Iterable[str]) -> List[str]:
    fields = set(BASE_ONLINE_FEATURE_ALLOWLIST)
    for key in row_keys:
        if key.startswith("online_gate_diff_sketch_"):
            fields.add(key)
        if key.startswith("online_prev_residual_sketch_"):
            fields.add(key)
        if key.startswith("online_fd_"):
            fields.add(key)
        if (
            key.startswith("online_ffro_")
            and not key.endswith("_elapsed_ms_pre")
            and "_vel_" not in key
            and "_step_" not in key
        ):
            fields.add(key)
    return sorted(fields)


def _extra_compute_observer_allowlist(row_keys: Iterable[str]) -> List[str]:
    """Features requiring additional forward compute, separated from online gates."""
    return sorted({
        key for key in row_keys
        if key.startswith("online_shadow_d")
        or (
            key.startswith("online_ffro_")
            and ("_vel_" in key or "_step_" in key or key.endswith("_elapsed_ms_pre"))
        )
    })


def _log_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.log(float(a) + EPS) - math.log(float(b) + EPS)


def _gate_sequence(rows: List[Dict[str, Any]]) -> str:
    return "".join("C" if bool(r.get("is_cached", False)) else "F" for r in rows)


def _non_pollution_replay_check(
    pipe,
    ctx: Dict[str, Any],
    fork_steps: List[int],
    native_final: torch.Tensor,
    native_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    replay_final, replay_rows, _ = _native_with_snapshots(pipe, ctx, fork_steps)
    native_f = native_final.detach().to(torch.float32)
    replay_f = replay_final.detach().to(native_f.device, dtype=torch.float32)
    seq_a = _gate_sequence(native_rows)
    seq_b = _gate_sequence(replay_rows)
    return {
        "enabled": True,
        "gate_sequence_equal": seq_a == seq_b,
        "row_count_equal": len(native_rows) == len(replay_rows),
        "final_l2": _norm(replay_f - native_f),
        "native_gate_sequence": seq_a,
        "replay_gate_sequence": seq_b,
    }


def _run_branch(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
    *,
    branch_action: str,
    tail: str,
    native_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    rows: List[Dict[str, Any]] = []
    start = int(snap["step_index"])
    with torch.no_grad():
        for i in range(start, len(ctx["timesteps"])):
            if i == start:
                forced = branch_action
            elif tail == "FA":
                forced = "full"
            elif tail == "SL":
                forced = "cache" if bool(native_rows[i].get("is_cached", False)) else "full"
            elif tail == "RE":
                forced = None
            else:
                raise ValueError(f"unsupported tail: {tail}")
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=forced)
            rows.append({
                "step_index": i,
                "forced_action": forced,
                "decision_u": rec["decision_u"],
                "native_decision_u": rec.get("native_decision_u"),
                "decision_reason": rec["decision_reason"],
                "is_cached": bool(rec["is_cached"]),
            })
    return {"final": latents.detach(), "rows": rows}


def _candidate_action_defect(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
) -> Dict[str, Any]:
    step = int(snap["step_index"])
    try:
        latents = _restore(pipe, snap)
        z_full_next, rec_full = _one_step(pipe, ctx, latents, step, forced_action="full")
        v_cf = rec_full["o_drv"].detach().to(torch.float32)
        latents = _restore(pipe, snap)
        z_cache_next, rec_cache = _one_step(pipe, ctx, latents, step, forced_action="cache")
        v_candidate = rec_cache["o_drv"].detach().to(torch.float32)
    except RuntimeError as exc:
        return {
            "candidate_valid": False,
            "candidate_error": repr(exc),
            "v_cf_norm": None,
            "forced_candidate_velocity_norm": None,
            "forced_candidate_action_defect_norm": None,
            "forced_candidate_action_defect_rel_to_v_cf": None,
            "forced_candidate_action_defect_cos_v_cf": None,
            "integrator_scaled_defect_norm": None,
            "integrator_scaled_defect_rel_to_full_step": None,
        }
    defect = v_candidate - v_cf
    step_defect = z_cache_next.detach().to(torch.float32) - z_full_next.detach().to(torch.float32)
    full_step_norm = _norm(z_full_next.detach().to(torch.float32) - snap["latents"].detach().to(torch.float32))
    return {
        "candidate_valid": True,
        "candidate_error": "",
        "v_cf_definition": "forced-full velocity on native cached fork state",
        "forced_candidate_action": "cache",
        "forced_candidate_action_defect_definition": "v_cache_candidate - v_cf",
        "v_cf_norm": _norm(v_cf),
        "forced_candidate_velocity_norm": _norm(v_candidate),
        "forced_candidate_action_defect_norm": _norm(defect),
        "forced_candidate_action_defect_rel_to_v_cf": _norm(defect) / (_norm(v_cf) + EPS),
        "forced_candidate_action_defect_cos_v_cf": (
            float(torch.sum(defect * v_cf).item()) / (_norm(defect) * _norm(v_cf) + EPS)
        ),
        "integrator_scaled_defect_definition": "scheduler_step(z, v_cache_candidate) - scheduler_step(z, v_cf)",
        "integrator_scaled_defect_norm": _norm(step_defect),
        "integrator_scaled_defect_rel_to_full_step": _norm(step_defect) / (full_step_norm + EPS),
    }


def _label_triplet(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
    native_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tail in ("FA", "SL", "RE"):
        try:
            branch_full = _run_branch(pipe, ctx, snap, branch_action="full", tail=tail, native_rows=native_rows)
            branch_cache = _run_branch(pipe, ctx, snap, branch_action="cache", tail=tail, native_rows=native_rows)
        except RuntimeError as exc:
            out[f"Y_{tail}_valid"] = False
            out[f"Y_{tail}_error"] = repr(exc)
            out[f"Y_{tail}_l2"] = None
            out[f"Y_{tail}_future_action_hamming"] = None
            continue
        z_f = branch_full["final"].to(torch.float32)
        z_c = branch_cache["final"].to(z_f.device, dtype=torch.float32)
        seq_f = ["C" if r["is_cached"] else "F" for r in branch_full["rows"]]
        seq_c = ["C" if r["is_cached"] else "F" for r in branch_cache["rows"]]
        out.update({
            f"Y_{tail}_valid": True,
            f"Y_{tail}_error": "",
            f"Y_{tail}_l2": _norm(z_c - z_f),
            f"Y_{tail}_full_branch_cache_rate": float(sum(s == "C" for s in seq_f) / max(len(seq_f), 1)),
            f"Y_{tail}_cache_branch_cache_rate": float(sum(s == "C" for s in seq_c) / max(len(seq_c), 1)),
            f"Y_{tail}_future_action_hamming": _hamming(seq_c, seq_f),
        })
    out["log_Y_SL_minus_FA"] = _log_delta(out.get("Y_SL_l2"), out.get("Y_FA_l2"))
    out["log_Y_RE_minus_SL"] = _log_delta(out.get("Y_RE_l2"), out.get("Y_SL_l2"))
    return out


def _rows_for_prompt(pipe, prompt: str, prompt_id: int, seed: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
    native_final, native_rows, snapshots = _native_with_snapshots(pipe, ctx, args.fork_steps_resolved)
    full_final = full_trace["final"].detach()
    native_final_l2 = _norm(native_final.to(torch.float32) - full_final.to(native_final.device, dtype=torch.float32))
    rows: List[Dict[str, Any]] = []
    for fork_step, snap in sorted(snapshots.items()):
        base: Dict[str, Any] = {
            "schema": "online_proxy_e0_row.v1",
            "run_name": args.run_name,
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "mode": args.mode,
            "cache_threshold": float(_threshold_for_args(args)),
            "fork_step": int(fork_step),
            "step_index": int(fork_step),
            "timestep": float(ctx["timesteps"][fork_step].detach().to("cpu").item()),
            "sigma_n": float(ctx["sigmas"][fork_step]),
            "sigma_np1": float(ctx["sigmas"][fork_step + 1]),
            "step_size_H": float(ctx["sigmas"][fork_step + 1] - ctx["sigmas"][fork_step]),
            "native_action": snap.get("native_action"),
            "native_decision_reason": snap.get("native_decision_reason"),
            "native_final_l2_to_full": float(native_final_l2),
            "candidate_context": "closed_native_cached_state",
            "snapshot_cache_digest": snap.get("cache_state_digest"),
            "snapshot_scheduler_digest": snap.get("scheduler_state_digest"),
            "snapshot_rng_digest": snap.get("rng_state_digest"),
        }
        for key, value in snap.items():
            if key.startswith("online_"):
                base[key] = value
        row = {
            **base,
            **_candidate_action_defect(pipe, ctx, snap),
            **_label_triplet(pipe, ctx, snap, native_rows),
        }
        row["all_labels_valid"] = bool(
            row.get("candidate_valid")
            and row.get("Y_FA_valid")
            and row.get("Y_SL_valid")
            and row.get("Y_RE_valid")
        )
        rows.append(row)

    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prompt_dir / "native_rows.jsonl", native_rows)
    _write_csv(prompt_dir / "online_proxy_e0_rows.csv", rows)
    non_pollution = {"enabled": False}
    if args.non_pollution_check:
        non_pollution = _non_pollution_replay_check(pipe, ctx, args.fork_steps_resolved, native_final, native_rows)
        if not bool(non_pollution.get("gate_sequence_equal")) or float(non_pollution.get("final_l2", 0.0)) > 1e-4:
            raise RuntimeError(
                f"non-pollution replay failed prompt={prompt_id}: "
                f"gate_sequence_equal={non_pollution.get('gate_sequence_equal')}, "
                f"final_l2={non_pollution.get('final_l2')}"
            )
    (prompt_dir / "manifest.json").write_text(
        json.dumps({
            "schema": "online_proxy_e0_prompt_manifest.v1",
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "mode": args.mode,
            "num_steps": int(args.num_steps),
            "fork_steps": [int(x) for x in args.fork_steps_resolved],
            "fork_step_policy": str(args.fork_step_policy),
            "dense_all_step_labels": bool(args.fork_step_policy == "dense_all_steps"),
            "eligible_fork_steps": _eligible_fork_steps(args.num_steps, args.first_enhance),
            "cache_params": _cache_params_for_args(args),
            "online_direction_sketch_dims": int(args.online_direction_sketch_dims),
            "shadow_depths": [int(x) for x in args.shadow_depths_tuple],
            "shadow_on": str(args.shadow_on),
            "native_final_l2_to_full": float(native_final_l2),
            "non_pollution_check": non_pollution,
            "n_rows": len(rows),
            "n_valid": sum(1 for r in rows if r.get("all_labels_valid")),
            "complete": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.sencache_thresh_main is None:
        args.sencache_thresh_main = (
            float(args.sencache_thresh) if args.sencache_thresh is not None else 0.07
        )
    if args.mode == "SenCache" and args.sencache_sensitivity_path is None:
        raise SystemExit("--mode SenCache requires --sencache_sensitivity_path")
    if args.sencache_sensitivity_path is not None:
        load_sensitivity_table(args.sencache_sensitivity_path)
    args.fork_steps_resolved, args.fork_step_policy = _resolve_fork_steps(args)
    if not args.fork_steps_resolved:
        raise ValueError("no valid fork steps")
    args.shadow_depths_tuple = _parse_int_list(args.shadow_depths)
    selected_ids = _parse_prompt_ids(args.prompt_ids, args.prompt_id_file)
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
        mode=args.mode,
        threshold=_threshold_for_args(args),
        num_steps=args.num_steps,
        first_enhance=args.first_enhance,
        teacache_backbone=args.teacache_backbone,
        teacache_variant=args.teacache_variant,
        sencache_sensitivity_path=(None if args.sencache_sensitivity_path is None
                                   else str(args.sencache_sensitivity_path)),
        sencache_thresh_start=float(args.sencache_thresh_start),
        sencache_thresh_main=float(args.sencache_thresh_main),
        sencache_K=int(args.sencache_K),
        sencache_threshold_scale=args.sencache_threshold_scale,
        sencache_switch_ratio=float(args.sencache_switch_ratio),
        sencache_ret_steps=int(args.sencache_ret_steps),
        sencache_cutoff_steps=int(args.sencache_cutoff_steps),
        online_direction_sketch_dims=args.online_direction_sketch_dims,
        shadow_depths=args.shadow_depths_tuple,
        shadow_on=args.shadow_on,
    )
    all_rows: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []
    git_sha, git_dirty = _git_commit()
    try:
        for prompt_id, prompt in shard_records:
            prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
            if args.resume and (prompt_dir / "manifest.json").is_file():
                print(f"[shard {args.shard_idx}] prompt {prompt_id} complete, skip", flush=True)
                continue
            prompt_seed = seed_for(args.seed, prompt_id)
            t_prompt = time.perf_counter()
            rows = _rows_for_prompt(pipe, prompt, prompt_id, prompt_seed, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_prompt
            all_rows.extend(rows)
            per_image_records.append({"idx": int(prompt_id), "denoise_s": float(dt), "decode_s": 0.0})
            valid = sum(1 for r in rows if r.get("all_labels_valid"))
            print(f"[shard {args.shard_idx}] prompt {prompt_id} done in {dt:.1f}s valid={valid}/{len(rows)}",
                  flush=True)
    finally:
        teardown()

    shard_csv = args.output_dir / f"online_proxy_e0_rows_shard{args.shard_idx}of{args.shard_count}.csv"
    _write_csv(shard_csv, all_rows)
    row_keys = {k for row in all_rows for k in row.keys()}
    allowlist = _online_feature_allowlist(row_keys)
    extra_observer_allowlist = _extra_compute_observer_allowlist(row_keys)
    _write_json(args.output_dir / f"online_feature_allowlist_shard{args.shard_idx}of{args.shard_count}.json", {
        "schema": "online_proxy_e0_feature_allowlist.v1",
        "description": "Columns allowed as online predictors; oracle labels/diagnostics are excluded.",
        "feature_allowlist": allowlist,
        "forbidden_examples": [
            "v_cf_norm",
            "forced_candidate_action_defect_norm",
            "Y_FA_l2",
            "Y_SL_l2",
            "Y_RE_l2",
            "native_final_l2_to_full",
        ],
    })
    _write_json(args.output_dir / f"extra_compute_observer_allowlist_shard{args.shard_idx}of{args.shard_count}.json", {
        "schema": "online_proxy_e0_extra_compute_observer_allowlist.v1",
        "description": (
            "Columns visible only after optional extra compute at the decision state. "
            "They may be used for observer/proxy diagnostics but are not zero-extra-cost online gate features."
        ),
        "feature_allowlist": extra_observer_allowlist,
    })
    manifest = {
        "schema": "online_proxy_e0_shard_manifest.v1",
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "mode": args.mode,
        "cache_params": {"threshold": _threshold_for_args(args), **_cache_params_for_args(args)},
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "prompt_file": str(args.prompt_file),
        "limit": int(args.limit),
        "selected_prompt_ids": [int(x) for x in selected_ids],
        "fork_steps": [int(x) for x in args.fork_steps_resolved],
        "fork_step_policy": str(args.fork_step_policy),
        "dense_all_step_labels": bool(args.fork_step_policy == "dense_all_steps"),
        "eligible_fork_steps": _eligible_fork_steps(args.num_steps, args.first_enhance),
        "online_direction_sketch_dims": int(args.online_direction_sketch_dims),
        "shadow_depths": [int(x) for x in args.shadow_depths_tuple],
        "shadow_on": str(args.shadow_on),
        "non_pollution_check": bool(args.non_pollution_check),
        "feature_allowlist_file": f"online_feature_allowlist_shard{args.shard_idx}of{args.shard_count}.json",
        "extra_compute_observer_allowlist_file": (
            f"extra_compute_observer_allowlist_shard{args.shard_idx}of{args.shard_count}.json"
        ),
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "n_rows": int(len(all_rows)),
        "n_valid": int(sum(1 for r in all_rows if r.get("all_labels_valid"))),
    }
    _write_json(args.output_dir / f"manifest_shard{args.shard_idx}of{args.shard_count}.json", manifest)
    write_timing_json(
        args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json",
        per_image=per_image_records,
        config={
            "cache_mode": f"online_proxy_e0_{args.mode}",
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
