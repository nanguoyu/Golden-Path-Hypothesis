#!/usr/bin/env python3
"""Geometric-vector impulse probe for FLUX cache error propagation.

This runner implements the first executable slice of
``docs/research_plan_geometric_vector_propagation.md``.  It does not deploy a
new gate.  It measures whether equal-norm velocity perturbations at a fixed
step have direction-dependent future propagation.

Two reference contexts are supported:

* ``clean``: the reference state is the full trajectory state ``z_k`` with
  cache memory built by forcing full steps before ``k``.
* ``closed``: the reference state is the native cached trajectory state
  ``tilde z_k`` saved immediately before ``k``.

For each context, the runner computes a reference full-after endpoint, injects
velocity perturbations of equal norm in selected directions, then measures the
terminal latent drift relative to that context's own full-after endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.a_k_probe import _encode_and_prepare  # noqa: E402
from flux.causal_fork_runner import _native_with_snapshots, _one_step, _restore, _snapshot  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX geometric-vector impulse probe.")
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
    p.add_argument("--fork_steps", default="6,15,18,25,35,45")
    p.add_argument("--fork_step_file", type=Path, default=None)
    p.add_argument("--contexts", default="clean,closed",
                   help="Comma/space separated subset of: clean, closed.")
    p.add_argument("--direction_types", default="actual,negative-actual,random,orthogonal-random",
                   help="Comma/space separated: actual, negative-actual, random, orthogonal-random.")
    p.add_argument("--random_dirs", type=int, default=8)
    p.add_argument("--direction_seed", type=int, default=0)
    p.add_argument("--epsilon_mode", choices=["actual", "absolute", "table"], default="actual",
                   help=(
                       "Velocity perturbation norm source. 'actual' uses "
                       "epsilon_scale * ||a_k|| for the current prompt/context; "
                       "'absolute' uses epsilon_abs; 'table' uses per-step values "
                       "from --epsilon_table multiplied by epsilon_scale."
                   ))
    p.add_argument("--epsilon_table", type=Path, default=None,
                   help=(
                       "Optional JSON/CSV table for --epsilon_mode table. JSON may "
                       "be {'6': value} or {'closed:6': value}; CSV needs columns "
                       "step_index, epsilon and optional context."
                   ))
    p.add_argument("--epsilon_scale", type=float, default=1.0,
                   help="Velocity perturbation norm multiplier relative to the actual cache velocity error norm.")
    p.add_argument("--epsilon_abs", type=float, default=0.0,
                   help="Fallback absolute velocity perturbation norm if an actual direction is unavailable.")
    p.add_argument("--non_pollution_check", action="store_true",
                   help="Replay native rollout after all probe branches and compare it with the original native run.")
    p.add_argument("--prompt_ids", default=None)
    p.add_argument("--prompt_id_file", type=Path, default=None)
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


def _load_epsilon_table(path: Optional[Path]) -> Dict[Tuple[str, int], float]:
    """Load optional formal-run perturbation norms.

    The returned keys are ``(context, step)``.  Context may be ``"*"``, which
    applies to clean and closed states.  This lets formal runs use calibration
    medians without tying the perturbation scale to each prompt's own
    cache-error norm.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"epsilon table not found: {path}")
    out: Dict[Tuple[str, int], float] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.items()
        elif isinstance(payload, list):
            rows = []
            for row in payload:
                if not isinstance(row, dict):
                    raise ValueError(f"invalid epsilon table row: {row!r}")
                rows.append((f"{row.get('context', '*')}:{row['step_index']}", row["epsilon"]))
        else:
            raise ValueError(f"unsupported epsilon table JSON type: {type(payload).__name__}")
        for key, value in rows:
            key_s = str(key)
            if ":" in key_s:
                context, step_s = key_s.split(":", 1)
            else:
                context, step_s = "*", key_s
            out[(context, int(step_s))] = float(value)
        return out

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "step_index" not in row or "epsilon" not in row:
                raise ValueError("epsilon CSV must contain step_index and epsilon columns")
            out[(row.get("context") or "*", int(row["step_index"]))] = float(row["epsilon"])
    return out


def _parse_tokens(text: str, allowed: Sequence[str], *, name: str) -> List[str]:
    allowed_set = set(allowed)
    out: List[str] = []
    for tok in text.replace(",", " ").split():
        value = tok.strip()
        if not value:
            continue
        if value not in allowed_set:
            raise ValueError(f"unsupported {name}: {value!r}; allowed={sorted(allowed_set)}")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"empty {name}")
    return out


def _run_name_for_args(args: argparse.Namespace, selected_prompt_count: int) -> str:
    thresh = _threshold_for_args(args)
    method = "teacache" if args.mode == "TeaCache" else "seacache"
    t_tag = str(thresh).replace(".", "")
    limit_tag = f"n{args.limit}" if args.limit > 0 else "nfull"
    if selected_prompt_count:
        limit_tag += f"_sel{selected_prompt_count}"
    return f"gvimpulse_{method}_t{t_tag}_{limit_tag}_s{args.seed}_{args.num_steps}"


def _seed_for_direction(base: int, prompt_id: int, step: int, context: str, dtype: str, idx: int) -> int:
    payload = f"{base}|{prompt_id}|{step}|{context}|{dtype}|{idx}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _unit(x: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
    y = x.detach().to(torch.float32)
    n = float(torch.linalg.vector_norm(y).item())
    if not math.isfinite(n) or n <= EPS:
        return None, n
    return y / n, n


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a.detach().to(torch.float32) * b.detach().to(torch.float32)).item())


def _random_unit_like(ref: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    r = torch.randn(tuple(ref.shape), generator=gen, dtype=torch.float32)
    r = r.to(device=ref.device)
    u, n = _unit(r)
    if u is None:
        raise RuntimeError(f"random direction has zero norm, seed={seed}, norm={n}")
    return u


def _orthogonal_unit(ref: torch.Tensor, base: torch.Tensor, seed: int) -> Optional[torch.Tensor]:
    r = _random_unit_like(ref, seed)
    b = base.detach().to(torch.float32)
    r = r - _dot(r, b) * b
    u, _ = _unit(r)
    return u


def _forward_velocity(
    pipe,
    ctx: Dict[str, Any],
    latents: torch.Tensor,
    step_index: int,
    *,
    forced_action: Optional[str],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    timestep = ctx["timesteps"][step_index]
    tr = pipe.transformer
    tr._td_run_kind = "cached"
    tr._td_with_cf = False
    tr._td_force_action = forced_action
    try:
        with torch.no_grad():
            t_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
            velocity = pipe.transformer(
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
    if int(rec["step"]) != int(step_index):
        raise RuntimeError(f"velocity step mismatch: rec={rec['step']} expected={step_index}")
    return velocity.detach(), rec


def _rollout_full_tail(pipe, ctx: Dict[str, Any], latents: torch.Tensor, start_step: int) -> torch.Tensor:
    tr = pipe.transformer
    tr._td_run_kind = "full"
    tr._td_with_cf = False
    tr._td_force_action = None
    tr._td_step = int(start_step)
    out = latents
    with torch.no_grad():
        for i in range(start_step, len(ctx["timesteps"])):
            timestep = ctx["timesteps"][i]
            t_expanded = timestep.expand(out.shape[0]).to(out.dtype)
            velocity = pipe.transformer(
                hidden_states=out,
                timestep=t_expanded / 1000,
                guidance=ctx["guidance"],
                pooled_projections=ctx["pooled_prompt_embeds"],
                encoder_hidden_states=ctx["prompt_embeds"],
                txt_ids=ctx["text_ids"],
                img_ids=ctx["latent_image_ids"],
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            _ensure_step_index(pipe.scheduler, timestep)
            out = pipe.scheduler.step(velocity, timestep, out, return_dict=False)[0]
    return out.detach()


def _full_after_reference(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        latents = _restore(pipe, snap)
        pipe.transformer._td_run_kind = "cached"
        pipe.transformer._td_with_cf = False
        step = int(snap["step_index"])
        latents_post, rec = _one_step(pipe, ctx, latents, step, forced_action="full")
        final = _rollout_full_tail(pipe, ctx, latents_post, step + 1)
    return {
        "pre": latents.detach(),
        "post": latents_post.detach(),
        "final": final.detach(),
        "velocity": rec["o_drv"].detach(),
    }


def _injected_endpoint(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
    direction: torch.Tensor,
    epsilon: float,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        latents = _restore(pipe, snap)
        step = int(snap["step_index"])
        v_full, _ = _forward_velocity(pipe, ctx, latents, step, forced_action="full")
        timestep = ctx["timesteps"][step]
        _ensure_step_index(pipe.scheduler, timestep)
        injected_velocity = v_full + float(epsilon) * direction.to(device=v_full.device, dtype=v_full.dtype)
        latents_post = pipe.scheduler.step(injected_velocity, timestep, latents, return_dict=False)[0]
        final = _rollout_full_tail(pipe, ctx, latents_post, step + 1)
    return {
        "post": latents_post.detach(),
        "final": final.detach(),
        "velocity_full": v_full.detach(),
        "velocity_injected": injected_velocity.detach(),
    }


def _actual_velocity_error(
    pipe,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    step = int(snap["step_index"])
    try:
        latents = _restore(pipe, snap)
        pipe.transformer._td_run_kind = "cached"
        pipe.transformer._td_with_cf = False
        v_full, rec_full = _forward_velocity(pipe, ctx, latents, step, forced_action="full")
        latents = _restore(pipe, snap)
        pipe.transformer._td_run_kind = "cached"
        pipe.transformer._td_with_cf = False
        v_cache, rec_cache = _forward_velocity(pipe, ctx, latents, step, forced_action="cache")
    except RuntimeError as exc:
        return None, {"valid": False, "error": repr(exc)}
    return v_cache.detach().to(torch.float32) - v_full.detach().to(torch.float32), {
        "valid": True,
        "error": "",
        "full_decision_reason": rec_full.get("decision_reason"),
        "cache_decision_reason": rec_cache.get("decision_reason"),
        "native_decision": rec_cache.get("native_decision_u"),
        "native_decision_reason": rec_cache.get("native_decision_reason"),
    }


def _clean_full_snapshots(
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
    with torch.no_grad():
        for i in range(len(ctx["timesteps"])):
            if i in wanted:
                snapshots[i] = _snapshot(pipe, latents, i)
                snapshots[i]["native_action"] = "clean-forced-full"
                snapshots[i]["native_decision_reason"] = "clean_forced_full_prefix"
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action="full")
            rows.append({
                "step_index": i,
                "decision_u": rec["decision_u"],
                "is_cached": bool(rec["is_cached"]),
                "decision_reason": rec["decision_reason"],
            })
    return latents.detach(), rows, snapshots


def _attach_cache_age(snapshots: Dict[int, Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    for step, snap in snapshots.items():
        age = 0
        j = int(step) - 1
        while j >= 0:
            if not bool(rows[j].get("is_cached", False)):
                break
            age += 1
            j -= 1
        snap["cache_age"] = int(age)


def _direction_specs(
    *,
    ref_tensor: torch.Tensor,
    actual_unit: Optional[torch.Tensor],
    direction_types: List[str],
    random_dirs: int,
    seed_base: int,
    prompt_id: int,
    step: int,
    context: str,
) -> List[Tuple[str, int, int, Optional[torch.Tensor], str]]:
    specs: List[Tuple[str, int, int, Optional[torch.Tensor], str]] = []
    if "actual" in direction_types:
        seed = _seed_for_direction(seed_base, prompt_id, step, context, "actual", 0)
        if actual_unit is None:
            specs.append(("actual", 0, seed, None, "actual velocity error unavailable"))
        else:
            specs.append(("actual", 0, seed, actual_unit, ""))
    if "negative-actual" in direction_types:
        seed = _seed_for_direction(seed_base, prompt_id, step, context, "negative-actual", 0)
        if actual_unit is None:
            specs.append(("negative-actual", 0, seed, None, "actual velocity error unavailable"))
        else:
            specs.append(("negative-actual", 0, seed, -actual_unit, ""))
    if "random" in direction_types:
        for j in range(int(random_dirs)):
            seed = _seed_for_direction(seed_base, prompt_id, step, context, "random", j)
            specs.append(("random", j, seed, _random_unit_like(ref_tensor, seed), ""))
    if "orthogonal-random" in direction_types:
        for j in range(int(random_dirs)):
            seed = _seed_for_direction(seed_base, prompt_id, step, context, "orthogonal-random", j)
            if actual_unit is None:
                specs.append(("orthogonal-random", j, seed, None, "actual velocity error unavailable"))
            else:
                u = _orthogonal_unit(ref_tensor, actual_unit, seed)
                specs.append(("orthogonal-random", j, seed, u,
                              "" if u is not None else "orthogonal direction degenerate"))
    return specs


def _epsilon_for_step(
    args: argparse.Namespace,
    *,
    context: str,
    step: int,
    actual_norm: float,
) -> Tuple[float, str, str]:
    if args.epsilon_mode == "actual":
        epsilon = float(args.epsilon_scale) * float(actual_norm)
        if epsilon <= EPS and float(args.epsilon_abs) > 0.0:
            return float(args.epsilon_abs), "actual_fallback_abs", ""
        return epsilon, "actual", "" if epsilon > EPS else "actual velocity error norm is zero"
    if args.epsilon_mode == "absolute":
        epsilon = float(args.epsilon_abs)
        return epsilon, "absolute", "" if epsilon > EPS else "epsilon_abs is zero"
    if args.epsilon_mode == "table":
        table: Dict[Tuple[str, int], float] = getattr(args, "epsilon_table_resolved", {})
        value = table.get((context, int(step)), table.get(("*", int(step))))
        if value is None:
            if float(args.epsilon_abs) > 0.0:
                return float(args.epsilon_abs), "table_missing_fallback_abs", ""
            return 0.0, "table_missing", f"epsilon table has no entry for {context}:{step}"
        epsilon = float(args.epsilon_scale) * float(value)
        return epsilon, "table", "" if epsilon > EPS else "epsilon table value is zero"
    raise ValueError(f"unsupported epsilon_mode: {args.epsilon_mode}")


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


def _run_context_rows(
    pipe,
    ctx: Dict[str, Any],
    *,
    args: argparse.Namespace,
    prompt_id: int,
    seed: int,
    context: str,
    snapshots: Dict[int, Dict[str, Any]],
    full_final: torch.Tensor,
    clean_final_l2: Optional[float],
    native_final_l2: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step, snap in sorted(snapshots.items()):
        ref: Optional[Dict[str, torch.Tensor]] = None
        try:
            ref = _full_after_reference(pipe, ctx, snap)
        except RuntimeError as exc:
            rows.append({
                "valid": False,
                "error": f"reference_full_after_failed: {exc!r}",
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "context": context,
                "step_index": int(step),
                "method": args.mode,
                "threshold": float(_threshold_for_args(args)),
            })
            continue

        actual_vec, actual_meta = _actual_velocity_error(pipe, ctx, snap)
        actual_unit, actual_norm = _unit(actual_vec) if actual_vec is not None else (None, 0.0)
        epsilon, epsilon_source, epsilon_error = _epsilon_for_step(
            args, context=context, step=int(step), actual_norm=float(actual_norm)
        )

        specs = _direction_specs(
            ref_tensor=ref["velocity"],
            actual_unit=actual_unit,
            direction_types=args.direction_types_resolved,
            random_dirs=args.random_dirs,
            seed_base=args.direction_seed,
            prompt_id=prompt_id,
            step=step,
            context=context,
        )
        for direction_type, direction_index, direction_seed, direction, direction_error in specs:
            base: Dict[str, Any] = {
                "valid": False,
                "error": direction_error or epsilon_error,
                "run_name": args.run_name,
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "context": context,
                "method": args.mode,
                "threshold": float(_threshold_for_args(args)),
                "step_index": int(step),
                "timestep": float(ctx["timesteps"][step].detach().to("cpu").item()),
                "sigma_n": float(ctx["sigmas"][step]),
                "sigma_np1": float(ctx["sigmas"][step + 1]),
                "step_size_h": float(_step_size(ctx, step)),
                "reference_endpoint_type": "full-terminal" if context == "clean" else "closed-full-after",
                "direction_type": direction_type,
                "direction_index": int(direction_index),
                "direction_seed": int(direction_seed),
                "epsilon": float(epsilon),
                "epsilon_mode": args.epsilon_mode,
                "epsilon_source": epsilon_source,
                "epsilon_scale": float(args.epsilon_scale),
                "epsilon_abs": float(args.epsilon_abs),
                "velocity_error_norm": float(actual_norm),
                "native_action": snap.get("native_action"),
                "native_decision_reason": snap.get("native_decision_reason"),
                "cache_age": snap.get("cache_age"),
                "actual_velocity_error_valid": bool(actual_meta.get("valid", False)),
                "actual_velocity_error_error": actual_meta.get("error", ""),
                "clean_prefix_final_l2": clean_final_l2,
                "native_final_l2": native_final_l2,
                "num_steps": int(args.num_steps),
                "git_commit": getattr(args, "git_commit", None),
                "git_dirty": getattr(args, "git_dirty", None),
                "cache_state_digest": snap.get("cache_state_digest"),
                "scheduler_state_digest": snap.get("scheduler_state_digest"),
                "rng_state_digest": snap.get("rng_state_digest"),
            }
            if epsilon_error:
                rows.append(base)
                continue
            if direction is None:
                rows.append(base)
                continue
            if epsilon <= EPS:
                rows.append({**base, "error": "epsilon is zero; provide actual direction or --epsilon_abs"})
                continue
            try:
                inj = _injected_endpoint(pipe, ctx, snap, direction, epsilon)
            except RuntimeError as exc:
                rows.append({**base, "error": f"injection_failed: {exc!r}"})
                continue
            final_drift = _norm(inj["final"].to(torch.float32) - ref["final"].to(inj["final"].device, dtype=torch.float32))
            injected_update_norm = _norm(inj["post"].to(torch.float32) - ref["post"].to(inj["post"].device, dtype=torch.float32))
            nominal_update_norm = abs(_step_size(ctx, step)) * float(epsilon)
            rows.append({
                **base,
                "valid": True,
                "error": "",
                "reference_final_to_full_l2": _norm(ref["final"].to(torch.float32) - full_final.to(ref["final"].device, dtype=torch.float32)),
                "final_latent_drift": float(final_drift),
                "injected_update_norm": float(injected_update_norm),
                "actual_update_norm": float(injected_update_norm),
                "nominal_update_norm": float(nominal_update_norm),
                "gain_by_update": float(final_drift / (injected_update_norm + EPS)),
                "gain_actual_update": float(final_drift / (injected_update_norm + EPS)),
                "gain_by_velocity_epsilon": float(final_drift / (abs(_step_size(ctx, step)) * epsilon + EPS)),
                "gain_nominal_update": float(final_drift / (nominal_update_norm + EPS)),
                "velocity_full_norm": _norm(ref["velocity"]),
                "base_velocity_norm": _norm(ref["velocity"]),
                "injected_velocity_norm": _norm(inj["velocity_injected"]),
                "direction_norm": _norm(direction),
            })
    return rows


def _step_size(ctx: Dict[str, Any], step: int) -> float:
    sigmas = [float(x) for x in ctx["sigmas"]]
    return float(sigmas[step + 1] - sigmas[step])


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
    clean_final, clean_rows, clean_snapshots = _clean_full_snapshots(pipe, ctx, args.fork_steps_resolved)
    native_final, native_rows, native_snapshots = _native_with_snapshots(pipe, ctx, args.fork_steps_resolved)
    _attach_cache_age(clean_snapshots, clean_rows)
    _attach_cache_age(native_snapshots, native_rows)

    full_final = full_trace["final"].detach()
    clean_final_l2 = _norm(clean_final.to(torch.float32) - full_final.to(clean_final.device, dtype=torch.float32))
    native_final_l2 = _norm(native_final.to(torch.float32) - full_final.to(native_final.device, dtype=torch.float32))

    rows: List[Dict[str, Any]] = []
    if "clean" in args.contexts_resolved:
        rows.extend(_run_context_rows(
            pipe, ctx, args=args, prompt_id=prompt_id, seed=seed, context="clean",
            snapshots=clean_snapshots, full_final=full_final,
            clean_final_l2=clean_final_l2, native_final_l2=native_final_l2,
        ))
    if "closed" in args.contexts_resolved:
        rows.extend(_run_context_rows(
            pipe, ctx, args=args, prompt_id=prompt_id, seed=seed, context="closed",
            snapshots=native_snapshots, full_final=full_final,
            clean_final_l2=clean_final_l2, native_final_l2=native_final_l2,
        ))

    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(full_final.detach().to("cpu", dtype=torch.bfloat16), prompt_dir / "baseline.pt")
    torch.save(native_final.detach().to("cpu", dtype=torch.bfloat16), prompt_dir / "native_cached.pt")
    _write_jsonl(prompt_dir / "geometric_impulse_rows.jsonl", rows)
    _write_jsonl(prompt_dir / "clean_prefix_rows.jsonl", clean_rows)
    _write_jsonl(prompt_dir / "native_rows.jsonl", native_rows)
    non_pollution = {"enabled": False}
    if args.non_pollution_check:
        non_pollution = _non_pollution_replay_check(
            pipe, ctx, args.fork_steps_resolved, native_final, native_rows
        )
        if not bool(non_pollution.get("gate_sequence_equal")) or float(non_pollution.get("final_l2", 0.0)) > 1e-4:
            raise RuntimeError(
                f"non-pollution replay failed prompt={prompt_id}: "
                f"gate_sequence_equal={non_pollution.get('gate_sequence_equal')}, "
                f"final_l2={non_pollution.get('final_l2')}"
            )
    valid_rows = [r for r in rows if r.get("valid") is True]
    prompt_manifest = {
        "prompt_id": int(prompt_id),
        "seed": int(seed),
        "mode": args.mode,
        "num_steps": int(args.num_steps),
        "fork_steps": [int(x) for x in args.fork_steps_resolved],
        "contexts": args.contexts_resolved,
        "direction_types": args.direction_types_resolved,
        "random_dirs": int(args.random_dirs),
        "epsilon_mode": args.epsilon_mode,
        "epsilon_table": str(args.epsilon_table) if args.epsilon_table else None,
        "epsilon_scale": float(args.epsilon_scale),
        "epsilon_abs": float(args.epsilon_abs),
        "cache_params": _cache_params_for_args(args),
        "clean_prefix_final_l2": clean_final_l2,
        "native_final_l2": native_final_l2,
        "non_pollution_check": non_pollution,
        "files": {
            "baseline_latent": "baseline.pt",
            "native_cached_latent": "native_cached.pt",
            "geometric_impulse_rows": "geometric_impulse_rows.jsonl",
            "clean_prefix_rows": "clean_prefix_rows.jsonl",
            "native_rows": "native_rows.jsonl",
        },
        "n_rows": len(rows),
        "n_valid": len(valid_rows),
        "complete": True,
    }
    (prompt_dir / "manifest.json").write_text(
        json.dumps(prompt_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    args = parse_args()
    args.fork_steps_resolved = [s for s in _parse_ints(args.fork_steps, args.fork_step_file)
                                if 0 <= s < int(args.num_steps)]
    if not args.fork_steps_resolved:
        raise ValueError("no valid fork steps")
    args.contexts_resolved = _parse_tokens(args.contexts, ["clean", "closed"], name="context")
    args.direction_types_resolved = _parse_tokens(
        args.direction_types,
        ["actual", "negative-actual", "random", "orthogonal-random"],
        name="direction type",
    )
    if int(args.random_dirs) < 0:
        raise ValueError("--random_dirs must be nonnegative")
    args.epsilon_table_resolved = _load_epsilon_table(args.epsilon_table)
    if args.epsilon_mode == "table" and not args.epsilon_table_resolved:
        raise ValueError("--epsilon_mode table requires a non-empty --epsilon_table")

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
    )
    all_rows: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []
    git_sha, git_dirty = _git_commit()
    args.git_commit = git_sha
    args.git_dirty = git_dirty
    try:
        for prompt_id, prompt in shard_records:
            prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
            if args.resume and (prompt_dir / "manifest.json").is_file():
                print(f"[shard {args.shard_idx}] prompt {prompt_id} complete, skip", flush=True)
                continue
            prompt_seed = seed_for(args.seed, prompt_id)
            t_prompt = time.perf_counter()
            rows = _run_prompt(pipe, prompt, prompt_id, prompt_seed, args)
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

    _write_csv(args.output_dir / f"geometric_impulse_rows_shard{args.shard_idx}of{args.shard_count}.csv", all_rows)
    manifest = {
        "schema": "geometric_impulse.v1",
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
        "contexts": args.contexts_resolved,
        "direction_types": args.direction_types_resolved,
        "random_dirs": int(args.random_dirs),
        "direction_seed": int(args.direction_seed),
        "epsilon_mode": args.epsilon_mode,
        "epsilon_table": str(args.epsilon_table) if args.epsilon_table else None,
        "epsilon_table_entries": int(len(args.epsilon_table_resolved)),
        "epsilon_scale": float(args.epsilon_scale),
        "epsilon_abs": float(args.epsilon_abs),
        "non_pollution_check": bool(args.non_pollution_check),
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
            "cache_mode": f"gv_impulse_{args.mode}",
            "num_steps": int(args.num_steps),
            "seed": int(args.seed),
            "shard_idx": int(args.shard_idx),
            "shard_count": int(args.shard_count),
        },
        model_load_s=model_load_end - t0,
        wallclock_total_s=time.perf_counter() - t0,
        device=device,
    )
    valid_rows = [r for r in all_rows if r.get("valid") is True]
    by_context: Dict[str, Dict[str, int]] = {}
    by_direction: Dict[str, Dict[str, int]] = {}
    for row in all_rows:
        ctx_name = str(row.get("context", "unknown"))
        direction_name = str(row.get("direction_type", "unknown"))
        for table, name in ((by_context, ctx_name), (by_direction, direction_name)):
            slot = table.setdefault(name, {"rows": 0, "valid": 0})
            slot["rows"] += 1
            if row.get("valid") is True:
                slot["valid"] += 1
    summary = {
        "schema": "geometric_impulse_summary.v1",
        "run_name": args.run_name,
        "n_rows": int(len(all_rows)),
        "n_valid": int(len(valid_rows)),
        "n_invalid": int(len(all_rows) - len(valid_rows)),
        "by_context": by_context,
        "by_direction": by_direction,
        "manifest": f"manifest_shard{args.shard_idx}of{args.shard_count}.json",
        "complete": True,
    }
    (args.output_dir / f"geometric_impulse_summary_shard{args.shard_idx}of{args.shard_count}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
