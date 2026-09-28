#!/usr/bin/env python3
"""Same-prefix action fork probe for fixed SeaCachePayload schedules.

The prefix follows a locked cache/full schedule.  At each requested fork step
we restore the exact same sampler/cache/history state twice, force one branch
to full and the other to cache, then continue with either the locked schedule
or all-full tail.  This isolates the marginal action value of Sea-only /
Sen-only schedule differences without changing the prefix state.
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
    _hamming,
    _one_step,
    _parse_ints,
    _restore,
    _snapshot,
)
from flux.trajectory_deviation_runner import (  # noqa: E402
    PAYLOAD_MODES,
    _git_commit,
    _load_payload_action_steps,
    _norm,
    _run_full_trace,
    install_trajectory_deviation,
    reset_td_state,
)
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--payload_schedule_dir", type=Path, required=True,
                   help="Fixed schedule decisions directory.")
    p.add_argument("--payload_mode", choices=sorted(PAYLOAD_MODES), default="reuse",
                   help="Payload used for cached steps in both action branches.")
    p.add_argument("--payload_blend", type=float, default=1.0)
    p.add_argument("--payload_sigma", type=float, default=0.5)
    p.add_argument("--payload_control_seed_salt", type=int, default=0)
    p.add_argument("--fork_steps", default="7,9,16,20,34,39,43,46")
    p.add_argument("--fork_step_file", type=Path, default=None)
    p.add_argument("--tail_policies", default="locked-after",
                   help="Comma/space separated: locked-after, full-after.")
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


def _selected_prompt_ids(args: argparse.Namespace) -> List[int]:
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
    thresh = str(float(args.seacache_thresh)).replace(".", "")
    limit_tag = f"n{args.limit}" if int(args.limit) > 0 else "nfull"
    if selected_count:
        limit_tag += f"_sel{selected_count}"
    return f"fixedactionfork_t{thresh}_{args.payload_mode}_{limit_tag}_s{args.seed}_{args.num_steps}"


def _tail_policies(text: str) -> List[str]:
    policies = [tok.strip() for tok in text.replace(",", " ").split() if tok.strip()]
    for policy in policies:
        if policy not in {"locked-after", "full-after"}:
            raise ValueError(f"unsupported tail policy: {policy}")
    return policies


def _native_fixed_with_snapshots(
    pipe,
    ctx: Dict[str, Any],
    *,
    action_steps: Set[int],
    fork_steps: Set[int],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    tr = pipe.transformer
    tr._td_run_kind = "cached"
    tr._td_with_cf = False
    reset_td_state(pipe, action_steps=action_steps)
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    snapshots: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i in range(len(ctx["timesteps"])):
            if i in fork_steps:
                snapshots[i] = _snapshot(pipe, latents, i)
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=None)
            row = {
                "step_index": int(i),
                "decision_u": rec.get("decision_u"),
                "is_cached": bool(rec.get("is_cached")),
                "decision_reason": rec.get("decision_reason"),
                "native_decision_u": rec.get("native_decision_u"),
                "schedule_locked": rec.get("schedule_locked"),
                "schedule_u": rec.get("schedule_u"),
                "payload_mode": rec.get("payload_mode"),
                "payload_used": rec.get("payload_used"),
                "payload_available": rec.get("payload_available"),
                "payload_fallback": rec.get("payload_fallback"),
                "cache_state_digest_before": rec.get("cache_state_digest_before"),
                "cache_state_digest_after": rec.get("cache_state_digest_after"),
                "scheduler_state_digest_before_cf": rec.get("scheduler_state_digest_before_cf"),
            }
            rows.append(row)
            if i in snapshots:
                snapshots[i]["native_action"] = rec.get("decision_u")
                snapshots[i]["native_decision_reason"] = rec.get("decision_reason")
                snapshots[i]["schedule_u"] = rec.get("schedule_u")
    return latents.detach(), rows, snapshots


def _run_branch(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
    *,
    action_steps: Set[int],
    branch_action: str,
    tail_policy: str,
) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    tr = pipe.transformer
    tr._td_payload_action_steps = set(int(x) for x in action_steps)
    rows: List[Dict[str, Any]] = []
    start = int(snap["step_index"])
    with torch.no_grad():
        for i in range(start, len(ctx["timesteps"])):
            if i == start:
                forced = branch_action
            elif tail_policy == "full-after":
                forced = "full"
            elif tail_policy == "locked-after":
                forced = None
            else:
                raise ValueError(f"unsupported tail policy: {tail_policy}")
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=forced)
            rows.append({
                "step_index": int(i),
                "forced_action": forced,
                "decision_u": rec.get("decision_u"),
                "native_decision_u": rec.get("native_decision_u"),
                "decision_reason": rec.get("decision_reason"),
                "is_cached": bool(rec.get("is_cached")),
                "schedule_locked": rec.get("schedule_locked"),
                "schedule_u": rec.get("schedule_u"),
                "payload_mode": rec.get("payload_mode"),
                "payload_used": rec.get("payload_used"),
                "payload_available": rec.get("payload_available"),
                "payload_fallback": rec.get("payload_fallback"),
                "cache_state_digest_before": rec.get("cache_state_digest_before"),
                "cache_state_digest_after": rec.get("cache_state_digest_after"),
            })
    return {"final": latents.detach(), "rows": rows}


def _branch_pair_rows(
    pipe,
    ctx: Dict[str, Any],
    full_final: torch.Tensor,
    native_final: torch.Tensor,
    native_rows: List[Dict[str, Any]],
    snapshots: Dict[int, Dict[str, Any]],
    *,
    action_steps: Set[int],
    prompt_id: int,
    seed: int,
    tail_policies: List[str],
    payload_mode: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    z_full_ref = full_final.to(torch.float32)
    z_native_ref = native_final.to(torch.float32)
    native_seq = ["C" if row.get("is_cached") else "F" for row in native_rows]
    for fork_step, snap in sorted(snapshots.items()):
        for tail_policy in tail_policies:
            base = {
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "fork_step": int(fork_step),
                "tail_policy": str(tail_policy),
                "payload_mode": str(payload_mode),
                "schedule_action": "cache" if int(fork_step) in action_steps else "full",
                "native_action": snap.get("native_action"),
                "native_decision_reason": snap.get("native_decision_reason"),
                "schedule_u": snap.get("schedule_u"),
                "snapshot_cache_digest": snap.get("cache_state_digest"),
                "snapshot_scheduler_digest": snap.get("scheduler_state_digest"),
                "snapshot_rng_digest": snap.get("rng_state_digest"),
            }
            try:
                branch_full = _run_branch(
                    pipe, ctx, snap,
                    action_steps=action_steps,
                    branch_action="full",
                    tail_policy=tail_policy,
                )
                branch_cache = _run_branch(
                    pipe, ctx, snap,
                    action_steps=action_steps,
                    branch_action="cache",
                    tail_policy=tail_policy,
                )
            except RuntimeError as exc:
                out.append({**base, "valid": False, "error": repr(exc)})
                continue

            z_f = branch_full["final"].to(torch.float32)
            z_c = branch_cache["final"].to(torch.float32)
            seq_f = ["C" if row.get("is_cached") else "F" for row in branch_full["rows"]]
            seq_c = ["C" if row.get("is_cached") else "F" for row in branch_cache["rows"]]
            native_suffix = native_seq[fork_step:]
            same_branch = branch_cache if int(fork_step) in action_steps else branch_full
            same_final = same_branch["final"].to(torch.float32)
            same_seq = ["C" if row.get("is_cached") else "F" for row in same_branch["rows"]]
            full_first = branch_full["rows"][0] if branch_full["rows"] else {}
            cache_first = branch_cache["rows"][0] if branch_cache["rows"] else {}
            out.append({
                **base,
                "valid": True,
                "error": "",
                "q_cache_final_drift": _norm(z_c - z_full_ref),
                "q_full_final_drift": _norm(z_f - z_full_ref),
                "delta_q_cache_minus_full": _norm(z_c - z_full_ref) - _norm(z_f - z_full_ref),
                "final_cache_full_l2": _norm(z_c - z_f),
                "native_fixed_final_drift": _norm(z_native_ref - z_full_ref),
                "native_noop_final_l2": (
                    _norm(same_final - z_native_ref) if tail_policy == "locked-after" else None
                ),
                "native_noop_action_hamming": (
                    _hamming(same_seq, native_suffix) if tail_policy == "locked-after" else None
                ),
                "future_action_hamming_cache_vs_full": _hamming(seq_c, seq_f),
                "full_branch_first_decision": full_first.get("decision_u"),
                "cache_branch_first_decision": cache_first.get("decision_u"),
                "full_branch_first_payload_used": full_first.get("payload_used"),
                "cache_branch_first_payload_used": cache_first.get("payload_used"),
                "full_branch_first_schedule_u": full_first.get("schedule_u"),
                "cache_branch_first_schedule_u": cache_first.get("schedule_u"),
                "cache_rate_cache_branch": float(sum(s == "C" for s in seq_c) / max(len(seq_c), 1)),
                "cache_rate_full_branch": float(sum(s == "C" for s in seq_f) / max(len(seq_f), 1)),
                "cache_rate_delta": float(
                    (sum(s == "C" for s in seq_c) - sum(s == "C" for s in seq_f)) / max(len(seq_c), 1)
                ),
            })
    return out


def _run_prompt(pipe, prompt: str, prompt_id: int, seed: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    action_steps = _load_payload_action_steps(
        args.payload_schedule_dir,
        prompt_id,
        expected_num_steps=int(args.num_steps),
        require_reuse_reference=False,
    )
    if action_steps is None:
        raise RuntimeError("fixed schedule loader returned no action steps")
    fork_steps = set(int(x) for x in args.fork_steps_resolved)
    full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
    native_final, native_rows, snapshots = _native_fixed_with_snapshots(
        pipe,
        ctx,
        action_steps=action_steps,
        fork_steps=fork_steps,
    )
    rows = _branch_pair_rows(
        pipe,
        ctx,
        full_trace["final"],
        native_final,
        native_rows,
        snapshots,
        action_steps=action_steps,
        prompt_id=prompt_id,
        seed=seed,
        tail_policies=args.tail_policies_resolved,
        payload_mode=str(args.payload_mode),
    )
    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prompt_dir / "native_fixed_rows.jsonl", native_rows)
    _write_csv(prompt_dir / "fixed_schedule_causal_fork_rows.csv", rows)
    (prompt_dir / "manifest.json").write_text(
        json.dumps({
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "mode": "SeaCachePayload",
            "payload_mode": str(args.payload_mode),
            "payload_schedule_dir": str(args.payload_schedule_dir),
            "num_steps": int(args.num_steps),
            "fork_steps": [int(x) for x in args.fork_steps_resolved],
            "tail_policies": args.tail_policies_resolved,
            "complete": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    args = parse_args()
    args.fork_steps_resolved = [
        s for s in _parse_ints(args.fork_steps, args.fork_step_file)
        if 0 <= s < int(args.num_steps)
    ]
    if not args.fork_steps_resolved:
        raise ValueError("no valid fork steps")
    args.tail_policies_resolved = _tail_policies(args.tail_policies)
    selected_ids = _selected_prompt_ids(args)
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
        mode="SeaCachePayload",
        threshold=float(args.seacache_thresh),
        num_steps=int(args.num_steps),
        first_enhance=int(args.first_enhance),
        payload_mode=str(args.payload_mode),
        payload_blend=float(args.payload_blend),
        payload_sigma=float(args.payload_sigma),
        payload_control_seed_salt=int(args.payload_control_seed_salt),
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

    _write_csv(
        args.output_dir / f"fixed_schedule_causal_fork_rows_shard{args.shard_idx}of{args.shard_count}.csv",
        all_rows,
    )
    git_sha, git_dirty = _git_commit()
    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "mode": "SeaCachePayload",
        "payload_mode": str(args.payload_mode),
        "payload_blend": float(args.payload_blend),
        "payload_sigma": float(args.payload_sigma),
        "payload_schedule_dir": str(args.payload_schedule_dir),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "prompt_file": str(args.prompt_file),
        "limit": int(args.limit),
        "selected_prompt_ids": [int(x) for x in selected_ids],
        "fork_steps": [int(x) for x in args.fork_steps_resolved],
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
            "cache_mode": "fixed_schedule_causal_fork_SeaCachePayload",
            "payload_mode": str(args.payload_mode),
            "payload_schedule_dir": str(args.payload_schedule_dir),
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
