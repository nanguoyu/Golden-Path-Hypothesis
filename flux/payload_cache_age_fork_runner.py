#!/usr/bin/env python3
"""Cache-age factorial payload fork probe for fixed-schedule SeaCachePayload.

This runner controls the fork latent/scheduler state and cache memory
separately.  It runs a forced-full prefix, saves cache/history memory after an
anchor step `a`, saves the full latent/scheduler state before a later fork step
`k`, then grafts the anchor memory onto the fork state and compares one forced
cached step with reuse vs forecast payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from flux.causal_fork_runner import _one_step, _parse_ints, _restore, _snapshot  # noqa: E402
from flux.payload_causal_fork_runner import (  # noqa: E402
    _branch_scalar_fields,
    _first_step_scalar_summary,
    _hamming,
    _tail_policies,
    _write_jsonl,
)
from flux.trajectory_deviation_runner import (  # noqa: E402
    PAYLOAD_MODES,
    _cache_params_for_args,
    _git_commit,
    _norm,
    _run_full_trace,
    _threshold_for_args,
    install_trajectory_deviation,
    reset_td_state,
)
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402


MEMORY_FIELDS = (
    "accumulated_rel_l1_distance",
    "previous_modulated_input",
    "previous_residual",
    "sencache_anchor_latent",
    "sencache_anchor_timestep",
    "sencache_anchor_step",
    "sencache_accumulated_skips",
    "history_fd_state",
    "online_previous_modulated_input_norm",
    "online_previous_residual_norm",
    "cache_state_digest",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--payload_mode", choices=sorted(PAYLOAD_MODES), default="taylor_o1",
                   help="Forecast/control payload for the forecast branch.")
    p.add_argument("--payload_blend", type=float, default=1.0)
    p.add_argument("--payload_sigma", type=float, default=0.5)
    p.add_argument("--tail_policies", default="full-after",
                   help="Comma/space separated tail policies; cache-age audit normally uses full-after.")
    p.add_argument("--anchor_steps", default="3,12,24,36",
                   help="Comma/space separated anchor steps.")
    p.add_argument("--anchor_step_file", type=Path, default=None)
    p.add_argument("--gaps", default="1,2,4,8,12",
                   help="Comma/space separated positive fork gaps k-a.")
    p.add_argument("--gap_file", type=Path, default=None)
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
    p.set_defaults(mode="SeaCachePayload", payload_schedule_dir=None)
    return p.parse_args()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _valid_anchor_gap_pairs(anchors: Sequence[int], gaps: Sequence[int], *, num_steps: int) -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    seen = set()
    for anchor in anchors:
        for gap in gaps:
            k = int(anchor) + int(gap)
            key = (int(anchor), int(gap), int(k))
            if key in seen:
                continue
            seen.add(key)
            if int(anchor) < 0 or int(gap) <= 0:
                continue
            if int(anchor) >= int(num_steps) - 1:
                continue
            if not (0 <= k < int(num_steps) - 1):
                continue
            pairs.append((int(anchor), int(gap), int(k)))
    return pairs


def _run_name_for_args(args: argparse.Namespace, selected_count: int) -> str:
    threshold = str(float(args.seacache_thresh)).replace(".", "")
    blend = str(float(args.payload_blend)).replace(".", "p")
    limit_tag = f"n{args.limit}" if int(args.limit) > 0 else "nfull"
    if selected_count:
        limit_tag += f"_sel{selected_count}"
    return (
        f"payloadcacheage_t{threshold}_{args.payload_mode}_b{blend}_"
        f"{limit_tag}_s{args.seed}_{args.num_steps}"
    )


def _stable_id(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _forced_full_snapshots(
    pipe,
    ctx: Dict[str, Any],
    *,
    anchors: Set[int],
    fork_steps: Set[int],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    tr = pipe.transformer
    tr._td_run_kind = "cached"
    tr._td_with_cf = False
    tr._td_payload_mode = "reuse"
    tr._td_payload_blend = 1.0
    reset_td_state(pipe, action_steps=set())
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    anchor_snaps: Dict[int, Dict[str, Any]] = {}
    fork_snaps: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i in range(len(ctx["timesteps"])):
            if i in fork_steps:
                fork_snaps[i] = _snapshot(pipe, latents, i)
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action="full")
            rows.append({
                "step_index": int(i),
                "forced_action": "full",
                "decision_u": rec.get("decision_u"),
                "is_cached": bool(rec.get("is_cached")),
                "decision_reason": rec.get("decision_reason"),
                "schedule_locked": rec.get("schedule_locked"),
                "schedule_u": rec.get("schedule_u"),
                "payload_mode": rec.get("payload_mode"),
                "payload_used": rec.get("payload_used"),
                "payload_fallback": rec.get("payload_fallback"),
                "payload_available": rec.get("payload_available"),
            })
            if i in anchors:
                anchor_snaps[i] = _snapshot(pipe, latents, i + 1)
                anchor_snaps[i]["anchor_full_step_decision"] = rec.get("decision_u")
                anchor_snaps[i]["anchor_full_step_reason"] = rec.get("decision_reason")
    return latents.detach(), rows, anchor_snaps, fork_snaps


def _graft_anchor_memory(
    *,
    anchor_step: int,
    gap: int,
    fork_step: int,
    anchor_snap: Dict[str, Any],
    fork_snap: Dict[str, Any],
) -> Dict[str, Any]:
    snap = dict(fork_snap)
    for key in MEMORY_FIELDS:
        snap[key] = anchor_snap.get(key)
    snap["step_index"] = int(fork_step)
    snap["cnt"] = int(fork_snap["cnt"])
    snap["synthetic_anchor_step"] = int(anchor_step)
    snap["synthetic_gap"] = int(gap)
    snap["synthetic_fork_step"] = int(fork_step)
    snap["synthetic_prefix_id"] = _stable_id({
        "anchor_step": int(anchor_step),
        "gap": int(gap),
        "fork_step": int(fork_step),
        "anchor_cache_digest": anchor_snap.get("cache_state_digest"),
        "fork_scheduler_digest": fork_snap.get("scheduler_state_digest"),
        "fork_rng_digest": fork_snap.get("rng_state_digest"),
    })
    return snap


def _run_payload_branch(
    pipe,
    ctx: Dict[str, Any],
    full_trace: Dict[str, Any],
    snap: Dict[str, Any],
    *,
    action_steps: Set[int],
    payload_mode: str,
    payload_blend: float,
    tail_policy: str,
) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    tr = pipe.transformer
    tr._td_payload_action_steps = set(int(x) for x in action_steps)
    tr._td_payload_mode = str(payload_mode)
    tr._td_payload_blend = float(payload_blend)
    old_with_cf = bool(getattr(tr, "_td_with_cf", True))
    tr._td_with_cf = True
    rows: List[Dict[str, Any]] = []
    start = int(snap["step_index"])
    try:
        with torch.no_grad():
            for i in range(start, len(ctx["timesteps"])):
                if i == start:
                    forced = "cache"
                elif tail_policy == "full-after":
                    forced = "full"
                else:
                    raise ValueError(f"cache-age runner only supports full-after tail for now, got {tail_policy}")
                latents_pre = latents
                latents, rec = _one_step(pipe, ctx, latents, i, forced_action=forced)
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
    finally:
        tr._td_with_cf = old_with_cf
    return {"final": latents.detach(), "rows": rows}


def _paired_rows(
    pipe,
    ctx: Dict[str, Any],
    *,
    full_trace: Dict[str, Any],
    full_final: torch.Tensor,
    pairs: Sequence[Tuple[int, int, int]],
    anchor_snaps: Dict[int, Dict[str, Any]],
    fork_snaps: Dict[int, Dict[str, Any]],
    prompt_id: int,
    seed: int,
    payload_mode: str,
    payload_blend: float,
    tail_policies: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    z_full = full_final.to(torch.float32)
    for anchor_step, gap, fork_step in pairs:
        anchor_snap = anchor_snaps.get(int(anchor_step))
        fork_snap = fork_snaps.get(int(fork_step))
        for tail_policy in tail_policies:
            base = {
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "anchor_step": int(anchor_step),
                "gap": int(gap),
                "fork_step": int(fork_step),
                "tail_policy": str(tail_policy),
                "forecast_payload_mode": str(payload_mode),
                "forecast_payload_blend": float(payload_blend),
                "anchor_cache_digest": None if anchor_snap is None else anchor_snap.get("cache_state_digest"),
                "fork_cache_digest_before_graft": None if fork_snap is None else fork_snap.get("cache_state_digest"),
                "fork_scheduler_digest": None if fork_snap is None else fork_snap.get("scheduler_state_digest"),
            }
            if anchor_snap is None or fork_snap is None:
                out.append({**base, "valid": False, "error": "missing synthetic anchor or fork snapshot"})
                continue
            snap = _graft_anchor_memory(
                anchor_step=anchor_step,
                gap=gap,
                fork_step=fork_step,
                anchor_snap=anchor_snap,
                fork_snap=fork_snap,
            )
            base.update({
                "synthetic_prefix_id": snap.get("synthetic_prefix_id"),
                "snapshot_cache_digest": snap.get("cache_state_digest"),
                "snapshot_scheduler_digest": snap.get("scheduler_state_digest"),
                "snapshot_rng_digest": snap.get("rng_state_digest"),
            })
            action_steps = {int(fork_step)}
            try:
                reuse = _run_payload_branch(
                    pipe, ctx, full_trace, snap,
                    action_steps=action_steps,
                    payload_mode="reuse",
                    payload_blend=1.0,
                    tail_policy=tail_policy,
                )
                forecast = _run_payload_branch(
                    pipe, ctx, full_trace, snap,
                    action_steps=action_steps,
                    payload_mode=payload_mode,
                    payload_blend=payload_blend,
                    tail_policy=tail_policy,
                )
            except RuntimeError as exc:
                out.append({**base, "valid": False, "error": repr(exc)})
                continue

            z_reuse = reuse["final"].to(torch.float32)
            z_forecast = forecast["final"].to(torch.float32)
            reuse_seq = ["C" if row.get("is_cached") else "F" for row in reuse["rows"]]
            forecast_seq = ["C" if row.get("is_cached") else "F" for row in forecast["rows"]]
            reuse_first = reuse["rows"][0] if reuse["rows"] else {}
            forecast_first = forecast["rows"][0] if forecast["rows"] else {}
            reuse_drift = _norm(z_reuse - z_full)
            forecast_drift = _norm(z_forecast - z_full)
            out.append({
                **base,
                "valid": True,
                "error": "",
                "reuse_final_drift": reuse_drift,
                "forecast_final_drift": forecast_drift,
                "improvement_reuse_minus_forecast": reuse_drift - forecast_drift,
                "forecast_reuse_final_l2": _norm(z_forecast - z_reuse),
                "reuse_first_is_cached": bool(reuse_first.get("is_cached")),
                "forecast_first_is_cached": bool(forecast_first.get("is_cached")),
                "reuse_first_payload_used": reuse_first.get("payload_used"),
                "forecast_first_payload_used": forecast_first.get("payload_used"),
                "forecast_first_payload_available": forecast_first.get("payload_available"),
                "forecast_first_payload_fallback": forecast_first.get("payload_fallback"),
                **_first_step_scalar_summary(reuse_first, forecast_first),
                "future_action_hamming_forecast_vs_reuse": _hamming(forecast_seq, reuse_seq),
                "reuse_cache_rate_suffix": float(sum(x == "C" for x in reuse_seq) / max(len(reuse_seq), 1)),
                "forecast_cache_rate_suffix": float(sum(x == "C" for x in forecast_seq) / max(len(forecast_seq), 1)),
            })
    return out


def _run_prompt(pipe, prompt: str, prompt_id: int, seed: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
    pairs = list(args.anchor_gap_pairs_resolved)
    anchors = {int(a) for a, _, _ in pairs}
    forks = {int(k) for _, _, k in pairs}
    forced_full_final, forced_full_rows, anchor_snaps, fork_snaps = _forced_full_snapshots(
        pipe,
        ctx,
        anchors=anchors,
        fork_steps=forks,
    )
    rows = _paired_rows(
        pipe,
        ctx,
        full_trace=full_trace,
        full_final=full_trace["final"],
        pairs=pairs,
        anchor_snaps=anchor_snaps,
        fork_snaps=fork_snaps,
        prompt_id=prompt_id,
        seed=seed,
        payload_mode=args.payload_mode,
        payload_blend=float(args.payload_blend),
        tail_policies=args.tail_policies_resolved,
    )
    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prompt_dir / "forced_full_rows.jsonl", forced_full_rows)
    _write_csv(prompt_dir / "payload_cache_age_fork_rows.csv", rows)
    (prompt_dir / "manifest.json").write_text(
        json.dumps({
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "num_steps": int(args.num_steps),
            "forecast_payload_mode": str(args.payload_mode),
            "forecast_payload_blend": float(args.payload_blend),
            "anchors_requested": [int(x) for x in args.anchor_steps_resolved],
            "gaps_requested": [int(x) for x in args.gaps_resolved],
            "anchor_gap_pairs": [
                {"anchor_step": int(a), "gap": int(g), "fork_step": int(k)}
                for a, g, k in pairs
            ],
            "anchors_materialized": sorted(int(x) for x in anchor_snaps),
            "fork_steps_materialized": sorted(int(x) for x in fork_snaps),
            "tail_policies": args.tail_policies_resolved,
            "forced_full_final_drift_vs_full_trace": _norm(
                forced_full_final.to(torch.float32) - full_trace["final"].to(torch.float32)
            ),
            "complete": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    args = parse_args()
    if args.payload_mode == "reuse" and float(args.payload_blend) >= 1.0:
        raise ValueError("forecast branch payload must differ from reuse")
    args.anchor_steps_resolved = [
        int(step) for step in _parse_ints(args.anchor_steps, args.anchor_step_file)
        if 0 <= int(step) < int(args.num_steps) - 1
    ]
    args.gaps_resolved = [
        int(gap) for gap in _parse_ints(args.gaps, args.gap_file)
        if int(gap) > 0
    ]
    args.anchor_gap_pairs_resolved = _valid_anchor_gap_pairs(
        args.anchor_steps_resolved,
        args.gaps_resolved,
        num_steps=int(args.num_steps),
    )
    if not args.anchor_gap_pairs_resolved:
        raise ValueError("no valid anchor/gap pairs")
    args.tail_policies_resolved = _tail_policies(args.tail_policies)
    unsupported = [policy for policy in args.tail_policies_resolved if policy != "full-after"]
    if unsupported:
        raise ValueError(f"cache-age runner only supports full-after tail policies: {unsupported}")
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

    _write_csv(args.output_dir / f"payload_cache_age_fork_rows_shard{args.shard_idx}of{args.shard_count}.csv", all_rows)
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
        },
        "forecast_payload_mode": str(args.payload_mode),
        "forecast_payload_blend": float(args.payload_blend),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "prompt_file": str(args.prompt_file),
        "limit": int(args.limit),
        "selected_prompt_ids": [int(x) for x in selected_ids],
        "anchors": [int(x) for x in args.anchor_steps_resolved],
        "gaps": [int(x) for x in args.gaps_resolved],
        "anchor_gap_pairs": [
            {"anchor_step": int(a), "gap": int(g), "fork_step": int(k)}
            for a, g, k in args.anchor_gap_pairs_resolved
        ],
        "tail_policies": args.tail_policies_resolved,
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
            "cache_mode": "payload_cache_age_fork_SeaCachePayload",
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
