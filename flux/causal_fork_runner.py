#!/usr/bin/env python3
"""Strict closed-state causal fork probe for FLUX SeaCache/TeaCache.

This runner estimates a marginal action value by saving the exact native
cached sampler state before a fork step, restoring it twice, forcing one branch
to `full` and the other to `cache`, then continuing with a specified tail
policy. It is intentionally separate from the observational trajectory audit.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from flux.trajectory_deviation_runner import (  # noqa: E402
    EPS,
    _cache_params_for_args,
    _ensure_step_index,
    _git_commit,
    _norm,
    _parse_prompt_id_tokens,
    _rng_digest,
    _run_full_trace,
    _scheduler_digest,
    _state_digest,
    _threshold_for_args,
    install_trajectory_deviation,
    reset_td_state,
)
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402
from lib.history_fd_observer import clone_state as clone_history_fd_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX strict causal fork probe.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--mode", choices=["SeaCache", "TeaCache"], default="SeaCache")
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
    p.add_argument("--fork_steps", default="3,6,15,25,35,45",
                   help="Comma/space separated fork step indices.")
    p.add_argument("--fork_step_file", type=Path, default=None)
    p.add_argument("--tail_policies", default="native-after,full-after",
                   help="Comma/space separated: native-after, full-after.")
    p.add_argument("--prompt_ids", default=None)
    p.add_argument("--prompt_id_file", type=Path, default=None)
    p.add_argument("--online_direction_sketch_dims", type=int, default=0,
                   help=(
                       "Record cheap online direction sketches with this many "
                       "block-projection dimensions. 0 disables sketches."
                   ))
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _parse_ints(text: Optional[str], path: Optional[Path] = None) -> List[int]:
    ids: List[int] = []
    if text:
        ids.extend(_parse_prompt_id_tokens(text))
    if path:
        ids.extend(_parse_prompt_id_tokens(path.read_text(encoding="utf-8")))
    out: List[int] = []
    seen = set()
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        out.append(int(value))
    return sorted(out)


def _run_name_for_args(args: argparse.Namespace, selected_prompt_count: int) -> str:
    thresh = _threshold_for_args(args)
    method = "teacache" if args.mode == "TeaCache" else "seacache"
    limit_tag = f"n{args.limit}" if args.limit > 0 else "nfull"
    if selected_prompt_count:
        limit_tag += f"_sel{selected_prompt_count}"
    t_tag = str(thresh).replace(".", "")
    return f"causalfork_{method}_t{t_tag}_{limit_tag}_s{args.seed}_{args.num_steps}"


def _clone_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.detach().clone()


def _norm_optional(t: Optional[torch.Tensor]) -> Optional[float]:
    return None if t is None else _norm(t)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _snapshot(pipe, latents: torch.Tensor, step_index: int) -> Dict[str, Any]:
    tr = pipe.transformer
    device = latents.device
    previous_modulated_input = _clone_optional(getattr(tr, "previous_modulated_input", None))
    previous_residual = _clone_optional(getattr(tr, "previous_residual", None))
    sencache_anchor_latent = _clone_optional(getattr(tr, "_td_sencache_anchor_latent", None))
    history_fd_state = clone_history_fd_state(getattr(tr, "_td_history_fd_state", None))
    snap: Dict[str, Any] = {
        "step_index": int(step_index),
        "latents": latents.detach().clone(),
        "scheduler_step_index": getattr(pipe.scheduler, "_step_index", None),
        "rng_cpu": torch.get_rng_state().clone(),
        "rng_cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
        "cnt": int(getattr(tr, "cnt", 0)),
        "accumulated_rel_l1_distance": float(getattr(tr, "accumulated_rel_l1_distance", 0.0)),
        "previous_modulated_input": previous_modulated_input,
        "previous_residual": previous_residual,
        "sencache_anchor_latent": sencache_anchor_latent,
        "sencache_anchor_timestep": getattr(tr, "_td_sencache_anchor_timestep", None),
        "sencache_anchor_step": getattr(tr, "_td_sencache_anchor_step", None),
        "sencache_accumulated_skips": int(getattr(tr, "_td_sencache_accumulated_skips", 0)),
        "history_fd_state": history_fd_state,
        "online_previous_modulated_input_norm": _norm_optional(previous_modulated_input),
        "online_previous_residual_norm": _norm_optional(previous_residual),
        "cache_state_digest": _state_digest(tr),
        "scheduler_state_digest": _scheduler_digest(pipe.scheduler),
        "rng_state_digest": _rng_digest(device),
    }
    return snap


def _restore(pipe, snap: Dict[str, Any]) -> torch.Tensor:
    tr = pipe.transformer
    latents = snap["latents"].detach().clone()
    pipe.scheduler._step_index = snap["scheduler_step_index"]
    torch.set_rng_state(snap["rng_cpu"])
    if latents.device.type == "cuda" and snap.get("rng_cuda") is not None:
        torch.cuda.set_rng_state(snap["rng_cuda"], latents.device)
    tr.cnt = int(snap["cnt"])
    tr.accumulated_rel_l1_distance = float(snap["accumulated_rel_l1_distance"])
    tr.previous_modulated_input = _clone_optional(snap["previous_modulated_input"])
    tr.previous_residual = _clone_optional(snap["previous_residual"])
    tr._td_sencache_anchor_latent = _clone_optional(snap.get("sencache_anchor_latent"))
    tr._td_sencache_anchor_timestep = snap.get("sencache_anchor_timestep")
    tr._td_sencache_anchor_step = snap.get("sencache_anchor_step")
    tr._td_sencache_accumulated_skips = int(snap.get("sencache_accumulated_skips", 0))
    tr._td_history_fd_state = clone_history_fd_state(snap.get("history_fd_state"))
    tr._td_force_action = None
    return latents


def _one_step(
    pipe,
    ctx: Dict[str, Any],
    latents: torch.Tensor,
    step_index: int,
    forced_action: Optional[str],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    timestep = ctx["timesteps"][step_index]
    tr = pipe.transformer
    tr._td_force_action = forced_action
    try:
        t_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
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
    finally:
        tr._td_force_action = None
    rec = dict(tr._td_last)
    _ensure_step_index(pipe.scheduler, timestep)
    latents_next = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents_next, rec


def _native_with_snapshots(
    pipe,
    ctx: Dict[str, Any],
    fork_steps: List[int],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    tr = pipe.transformer
    tr._td_run_kind = "cached"
    tr._td_with_cf = False
    reset_td_state(pipe)
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    wanted = set(fork_steps)
    snapshots: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    sigmas = [float(x) for x in ctx["sigmas"]]
    last_refresh_step = 0
    cache_run_len = 0
    num_cached_so_far = 0
    c_stale = 0.0
    c_traj = 0.0
    c_mem = 0.0
    with torch.no_grad():
        for i in range(len(ctx["timesteps"])):
            if i in wanted:
                snapshots[i] = _snapshot(pipe, latents, i)
                H_i_pre = sigmas[i + 1] - sigmas[i] if i + 1 < len(sigmas) else 0.0
                snapshots[i].update({
                    "online_last_refresh_step_pre": int(last_refresh_step),
                    "online_cache_age_pre": int(i - last_refresh_step),
                    "online_cache_run_len_pre": int(cache_run_len),
                    "online_num_cached_so_far_pre": int(num_cached_so_far),
                    "online_cache_ratio_so_far_pre": float(num_cached_so_far / max(i, 1)),
                    "online_c_stale_pre": float(c_stale),
                    "online_c_traj_pre": float(c_traj),
                    "online_c_mem_pre": float(c_mem),
                    "online_step_size_abs": abs(float(H_i_pre)),
                })
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=None)
            H_i = sigmas[i + 1] - sigmas[i] if i + 1 < len(sigmas) else 0.0
            acc_before = _maybe_float(rec.get("gate_accumulator_before"))
            gate_inc = _maybe_float(rec.get("gate_increment_rescaled"))
            p_path_decision = (
                acc_before + gate_inc
                if acc_before is not None and gate_inc is not None
                else None
            )
            q_proxy_no_sa = (
                p_path_decision * abs(float(H_i))
                if p_path_decision is not None
                else None
            )
            native_is_cached = bool(rec["is_cached"])
            online_row_fields = {
                "online_last_refresh_step_pre": int(last_refresh_step),
                "online_cache_age_pre": int(i - last_refresh_step),
                "online_cache_run_len_pre": int(cache_run_len),
                "online_num_cached_so_far_pre": int(num_cached_so_far),
                "online_cache_ratio_so_far_pre": float(num_cached_so_far / max(i, 1)),
                "online_c_stale_pre": float(c_stale),
                "online_c_traj_pre": float(c_traj),
                "online_c_mem_pre": float(c_mem),
                "online_step_size_abs": abs(float(H_i)),
                "online_gate_increment_raw": rec.get("gate_increment_raw"),
                "online_gate_increment_rescaled": rec.get("gate_increment_rescaled"),
                "online_gate_accumulator_before": rec.get("gate_accumulator_before"),
                "online_gate_accumulator_after_commit": rec.get("gate_accumulator_after"),
                "online_threshold_margin_before": rec.get("threshold_margin_before"),
                "online_threshold_margin_after": rec.get("threshold_margin_after"),
                "online_p_path_decision": p_path_decision,
                "online_q_proxy_no_sa": q_proxy_no_sa,
                "online_native_action": rec["decision_u"],
                "online_native_is_cached": native_is_cached,
            }
            for key, value in rec.items():
                if key.startswith("online_"):
                    online_row_fields.setdefault(key, value)
            rows.append({
                "step_index": i,
                "decision_u": rec["decision_u"],
                "is_cached": bool(rec["is_cached"]),
                "decision_reason": rec["decision_reason"],
                "cache_state_digest_before": rec["cache_state_digest_before"],
                "cache_state_digest_after": rec["cache_state_digest_after"],
                "scheduler_state_digest_before_cf": rec["scheduler_state_digest_before_cf"],
                **online_row_fields,
            })
            if i in snapshots:
                snapshots[i]["native_action"] = rec["decision_u"]
                snapshots[i]["native_decision_reason"] = rec["decision_reason"]
                snapshots[i].update(online_row_fields)
            q_for_commit = float(q_proxy_no_sa or 0.0)
            old_traj = c_traj
            if native_is_cached:
                c_traj = old_traj + q_for_commit
                c_stale = c_stale + q_for_commit
                cache_run_len += 1
                num_cached_so_far += 1
            else:
                c_traj = old_traj
                c_stale = 0.0
                c_mem = old_traj
                cache_run_len = 0
                last_refresh_step = i
    return latents.detach(), rows, snapshots


def _run_branch(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
    *,
    branch_action: str,
    tail_policy: str,
) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    rows: List[Dict[str, Any]] = []
    start = int(snap["step_index"])
    with torch.no_grad():
        for i in range(start, len(ctx["timesteps"])):
            if i == start:
                forced = branch_action
            elif tail_policy == "full-after":
                forced = "full"
            elif tail_policy == "native-after":
                forced = None
            else:
                raise ValueError(f"unsupported tail policy: {tail_policy}")
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=forced)
            rows.append({
                "step_index": i,
                "forced_action": forced,
                "decision_u": rec["decision_u"],
                "native_decision_u": rec.get("native_decision_u"),
                "decision_reason": rec["decision_reason"],
                "is_cached": bool(rec["is_cached"]),
                "cache_state_digest_before": rec["cache_state_digest_before"],
                "cache_state_digest_after": rec["cache_state_digest_after"],
            })
    return {"final": latents.detach(), "rows": rows}


def _hamming(a: List[str], b: List[str]) -> int:
    return int(sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)))


def _branch_pair_rows(
    pipe,
    ctx: Dict[str, Any],
    full_final: torch.Tensor,
    native_final: torch.Tensor,
    native_rows: List[Dict[str, Any]],
    snapshots: Dict[int, Dict[str, Any]],
    *,
    prompt_id: int,
    seed: int,
    tail_policies: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fork_step, snap in sorted(snapshots.items()):
        for tail_policy in tail_policies:
            base: Dict[str, Any] = {
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "fork_step": int(fork_step),
                "tail_policy": tail_policy,
                "native_action": snap.get("native_action"),
                "native_decision_reason": snap.get("native_decision_reason"),
                "snapshot_cache_digest": snap["cache_state_digest"],
                "snapshot_scheduler_digest": snap["scheduler_state_digest"],
                "snapshot_rng_digest": snap["rng_state_digest"],
            }
            for key, value in snap.items():
                if key.startswith("online_"):
                    base[key] = value
            try:
                branch_full = _run_branch(pipe, ctx, snap, branch_action="full", tail_policy=tail_policy)
                branch_cache = _run_branch(pipe, ctx, snap, branch_action="cache", tail_policy=tail_policy)
            except RuntimeError as exc:
                out.append({**base, "valid": False, "error": repr(exc)})
                continue

            z_f = branch_full["final"].to(torch.float32)
            z_c = branch_cache["final"].to(torch.float32)
            z_full_ref = full_final.to(z_f.device, dtype=torch.float32)
            z_native_ref = native_final.to(z_f.device, dtype=torch.float32)
            seq_f = ["C" if r["is_cached"] else "F" for r in branch_full["rows"]]
            seq_c = ["C" if r["is_cached"] else "F" for r in branch_cache["rows"]]
            native_suffix = ["C" if r["is_cached"] else "F" for r in native_rows[fork_step:]]
            same_branch = branch_cache if snap.get("native_action") == "cache" else branch_full
            same_final = same_branch["final"].to(z_f.device, dtype=torch.float32)
            same_seq = ["C" if r["is_cached"] else "F" for r in same_branch["rows"]]
            out.append({
                **base,
                "valid": True,
                "error": "",
                "q_cache_final_drift": _norm(z_c - z_full_ref),
                "q_full_final_drift": _norm(z_f - z_full_ref),
                "delta_q_cache_minus_full": _norm(z_c - z_full_ref) - _norm(z_f - z_full_ref),
                "final_cache_full_l2": _norm(z_c - z_f),
                "native_noop_final_l2": _norm(same_final - z_native_ref) if tail_policy == "native-after" else None,
                "native_noop_action_hamming": _hamming(same_seq, native_suffix) if tail_policy == "native-after" else None,
                "future_action_hamming_cache_vs_full": _hamming(seq_c, seq_f),
                "cache_rate_cache_branch": float(sum(s == "C" for s in seq_c) / max(len(seq_c), 1)),
                "cache_rate_full_branch": float(sum(s == "C" for s in seq_f) / max(len(seq_f), 1)),
                "cache_rate_delta": float(
                    (sum(s == "C" for s in seq_c) - sum(s == "C" for s in seq_f)) / max(len(seq_c), 1)
                ),
                "full_branch_first_decision": branch_full["rows"][0]["decision_u"],
                "cache_branch_first_decision": branch_cache["rows"][0]["decision_u"],
            })
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
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


def _run_prompt(pipe, prompt: str, prompt_id: int, seed: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
    native_final, native_rows, snapshots = _native_with_snapshots(pipe, ctx, args.fork_steps_resolved)
    rows = _branch_pair_rows(
        pipe,
        ctx,
        full_trace["final"],
        native_final,
        native_rows,
        snapshots,
        prompt_id=prompt_id,
        seed=seed,
        tail_policies=args.tail_policies_resolved,
    )
    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prompt_dir / "native_rows.jsonl", native_rows)
    _write_csv(prompt_dir / "causal_fork_rows.csv", rows)
    (prompt_dir / "manifest.json").write_text(
        json.dumps({
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "mode": args.mode,
            "num_steps": int(args.num_steps),
            "fork_steps": [int(x) for x in args.fork_steps_resolved],
            "tail_policies": args.tail_policies_resolved,
            "cache_params": _cache_params_for_args(args),
            "complete": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    args = parse_args()
    args.fork_steps_resolved = [s for s in _parse_ints(args.fork_steps, args.fork_step_file)
                                if 0 <= s < int(args.num_steps)]
    if not args.fork_steps_resolved:
        raise ValueError("no valid fork steps")
    args.tail_policies_resolved = [
        tok.strip() for tok in args.tail_policies.replace(",", " ").split() if tok.strip()
    ]
    for policy in args.tail_policies_resolved:
        if policy not in {"native-after", "full-after"}:
            raise ValueError(f"unsupported tail policy: {policy}")

    selected_ids = _parse_ints(args.prompt_ids, args.prompt_id_file)
    if args.run_name is None:
        args.run_name = _run_name_for_args(args, len(selected_ids))
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
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
        online_direction_sketch_dims=args.online_direction_sketch_dims,
    )
    all_rows: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []
    try:
        for prompt_id, prompt in shard_records:
            prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
            if args.resume and (prompt_dir / "manifest.json").is_file():
                print(f"[shard {args.shard_idx}] prompt {prompt_id} complete, skip", flush=True)
                continue
            seed = seed_for(args.seed, prompt_id)
            t_prompt = time.perf_counter()
            rows = _run_prompt(pipe, prompt, prompt_id, seed, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_prompt
            all_rows.extend(rows)
            per_image_records.append({"idx": int(prompt_id), "denoise_s": float(dt), "decode_s": 0.0})
            valid = sum(1 for r in rows if r.get("valid") is True)
            print(f"[shard {args.shard_idx}] prompt {prompt_id} done in {dt:.1f}s valid={valid}/{len(rows)}",
                  flush=True)
    finally:
        teardown()

    _write_csv(args.output_dir / f"causal_fork_rows_shard{args.shard_idx}of{args.shard_count}.csv", all_rows)
    git_sha, git_dirty = _git_commit()
    manifest = {
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
        "tail_policies": args.tail_policies_resolved,
        "online_direction_sketch_dims": int(args.online_direction_sketch_dims),
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
            "cache_mode": f"causal_fork_{args.mode}",
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
