#!/usr/bin/env python3
"""Experiment 6: propagation closure test.

docs/research_plan_after_extension.md — added after Exp 0-5. The defect
layer is algebraically closed by construction (`d_real = d_clean + d_state`
one-block; `d_real = d_nonmem + d_mem` two-block). What is untested is the
*propagation*: does each per-step velocity defect, propagated through the
subsequent full computation, reconstruct the real interval error?

For step k the nonlinear propagation operator (Version B — no JVP) is

    P_{k->N}(d) = T_{k+1:N}(z_{k+1} + H_k d) - z_N^base

realised by *velocity injection*: the per-step velocity defect `d` is added
to the driving velocity at step k, so the solver applies the drift H_k d in
fp32 inside its own Euler step. Injecting H_k d as a latent perturbation
would quantise it — H_k ~ 7e-3 makes H_k d smaller than the latent's bf16
ulp, which costs ~half the perturbation to rounding.

The probe is *self-contained*: defects, E_real and propagation are all
produced by one instrumented forward per mode, so there is no cross-runner
inconsistency. A one-cached-step interval then closes exactly.

  mode oneblock:  forward = flux/instrumented_cache_runner._instr_forward.
                  the reference run (all full, memory frozen at a) yields
                  z_N^full and d_clean; the [a,b]-cached run yields z_cache
                  and d_real; d_state = d_real - d_clean.
  mode twoblock:  forward = flux/twoblock_probe._twoblock_forward.
                  the reference run yields the clean refresh memory m*_b;
                  the block-1-cached run yields z_N^{b1}; the block-1+2
                  run yields z_N^{b1+b2} and d_nonmem / d_mem.

  E_real     = oneblock: z_N^{cache [a,b]} - z_N^full
               twoblock: z_N^{cache b1+b2} - z_N^{cache b1 only}
  E_combined = sum_k P(d_real)
  E_sum      = sum_k [ P(d_component1) + P(d_component2) ]

`E_combined vs E_real` is the propagation-closure test; `E_sum vs
E_combined` isolates propagation nonlinearity (P(x+y) != P(x)+P(y)).

Output per prompt: prompt_XXXXX/closure_<tag>.pt
  {mode, tag, unit, E_real, E_sum, E_combined, per_step:[{k, norm_P_*}], ...}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.io_utils import read_prompts, split_shard  # noqa: E402
from flux.postblock_probe import run_trace  # noqa: E402
from flux.instrumented_cache_runner import install_instr  # noqa: E402
from flux.twoblock_probe import install_twoblock  # noqa: E402


def _by_step(dump: List[Dict], fields: Tuple[str, ...]
             ) -> Dict[int, Dict[str, torch.Tensor]]:
    """{step: {field: fp32 tensor}} from an instrumented-forward dump,
    skipping records where any requested field is None (e.g. v_cache before
    the freeze step)."""
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for r in dump:
        if any(r.get(f) is None for f in fields):
            continue
        out[int(r["step"])] = {f: r[f].float() for f in fields}
    return out


def _run_instr(pipe, prompt, seed, args, *, cache_steps, freeze_step,
               velocity_inject=None) -> Tuple[torch.Tensor, List[Dict]]:
    """Drive the Exp 2 instrumented forward; return (z_N, dump).

    `_instr_forward` always runs full blocks (so v_full is recorded at every
    step) and drives the trajectory with the cache velocity at `cache_steps`.
    `velocity_inject` adds a per-step defect to the driving velocity.
    """
    teardown = install_instr(pipe, cache_steps=list(cache_steps),
                             freeze_step=int(freeze_step),
                             num_steps=int(args.num_steps),
                             velocity_inject=velocity_inject)
    try:
        trace = run_trace(pipe, prompt, seed, args)
        dump = list(pipe.transformer.dump)
    finally:
        teardown()
    return trace[-1], dump


def _run_twoblock(pipe, prompt, seed, args, *, cache1, cache2, freeze1,
                  freeze2, m_clean_b, record_residual_steps=(),
                  velocity_inject=None
                  ) -> Tuple[torch.Tensor, List[Dict], Dict[int, torch.Tensor]]:
    """Drive the Exp 3 two-block forward; return (z_N, dump, residuals)."""
    teardown = install_twoblock(pipe, cache1=list(cache1), cache2=list(cache2),
                                freeze1=int(freeze1), freeze2=int(freeze2),
                                m_clean_b=m_clean_b,
                                record_residual_steps=list(record_residual_steps),
                                num_steps=int(args.num_steps),
                                velocity_inject=velocity_inject)
    try:
        trace = run_trace(pipe, prompt, seed, args)
        tr = pipe.transformer
        dump, residuals = list(tr.dump), dict(tr.residuals)
    finally:
        teardown()
    return trace[-1], dump, residuals


def _accumulate_and_save(prompt_dir: Path, mode: str, tag: str, unit,
                         N: int, defects: Dict[int, Dict[str, torch.Tensor]],
                         comp_keys: Tuple[str, str], e_real: torch.Tensor,
                         propagate) -> int:
    """Sum E_sum / E_combined over all cached steps and save closure_<tag>.pt.

    `propagate(k, d)` returns P_{k->N}(d) = z_N' - z_N^base. `comp_keys` are
    the two decomposition components; `d_real` is propagated separately.
    """
    steps = sorted(defects)
    e_sum = torch.zeros_like(e_real)
    e_combined = torch.zeros_like(e_real)
    per_step = []
    for k in steps:
        rec = {"k": k}
        for ck in comp_keys:
            P = propagate(k, defects[k][ck])
            e_sum = e_sum + P
            rec[f"norm_P_{ck}"] = float(P.norm())
        P_real = propagate(k, defects[k]["d_real"])
        e_combined = e_combined + P_real
        rec["norm_P_d_real"] = float(P_real.norm())
        per_step.append(rec)
    torch.save({
        "mode": mode, "tag": tag, "unit": list(unit),
        "num_steps": N, "n_steps": len(steps),
        "E_real": e_real.to("cpu", torch.bfloat16),
        "E_sum": e_sum.to("cpu", torch.bfloat16),
        "E_combined": e_combined.to("cpu", torch.bfloat16),
        "per_step": per_step,
    }, prompt_dir / f"closure_{tag}.pt")
    return len(steps)


def _oneblock_prompt(pipe, prompt, seed, args, units, prompt_dir, N) -> None:
    """All oneblock intervals for one prompt (reference runs cached by anchor)."""
    ref_cache: Dict[int, Tuple[torch.Tensor, Dict]] = {}
    for (a, b) in units:
        tag = f"a{a:02d}_b{b:02d}"
        if a not in ref_cache:
            z_full, ref_dump = _run_instr(pipe, prompt, seed, args,
                                          cache_steps=[], freeze_step=a)
            ref_cache[a] = (z_full, _by_step(ref_dump, ("v_full", "v_cache")))
        z_full, ref = ref_cache[a]
        z_cache, cac_dump = _run_instr(pipe, prompt, seed, args,
                                       cache_steps=range(a + 1, b), freeze_step=a)
        cac = _by_step(cac_dump, ("v_full", "v_cache"))
        defects = {}
        for k in range(a + 1, b):
            if k not in ref or k not in cac:
                continue
            d_clean = ref[k]["v_cache"] - ref[k]["v_full"]
            d_real = cac[k]["v_cache"] - cac[k]["v_full"]
            defects[k] = {"d_clean": d_clean, "d_state": d_real - d_clean,
                          "d_real": d_real}
        e_real = z_cache.float() - z_full.float()

        def propagate(k, d, _zr=z_full):
            zN, _ = _run_instr(pipe, prompt, seed, args, cache_steps=[],
                               freeze_step=N, velocity_inject={k: d})
            return zN.float() - _zr.float()

        _accumulate_and_save(prompt_dir, "oneblock", tag, (a, b), N,
                             defects, ("d_clean", "d_state"), e_real, propagate)


def _twoblock_prompt(pipe, prompt, seed, args, units, prompt_dir, N) -> None:
    """All twoblock configs for one prompt (one shared reference run)."""
    b_steps = sorted({b for (_, b, _) in units})
    _, _, residuals = _run_twoblock(pipe, prompt, seed, args, cache1=[],
                                    cache2=[], freeze1=0, freeze2=0,
                                    m_clean_b=None, record_residual_steps=b_steps)
    m_clean = {b: residuals[b] for b in b_steps}

    for (a, b, c) in units:
        tag = f"a{a:02d}_b{b:02d}_c{c:02d}"
        mcb = m_clean[b]
        z_b1, _, _ = _run_twoblock(pipe, prompt, seed, args,
                                   cache1=range(a + 1, b), cache2=[],
                                   freeze1=a, freeze2=b, m_clean_b=mcb)
        z_b1b2, dump, _ = _run_twoblock(pipe, prompt, seed, args,
                                        cache1=range(a + 1, b),
                                        cache2=range(b + 1, c),
                                        freeze1=a, freeze2=b, m_clean_b=mcb)
        dd = _by_step(dump, ("v_full", "v_cache_real", "v_cache_clean"))
        defects = {}
        for k, r in dd.items():
            d_nonmem = r["v_cache_clean"] - r["v_full"]
            d_mem = r["v_cache_real"] - r["v_cache_clean"]
            defects[k] = {"d_nonmem": d_nonmem, "d_mem": d_mem,
                          "d_real": d_nonmem + d_mem}
        e_real = z_b1b2.float() - z_b1.float()

        def propagate(k, d, _a=a, _b=b, _mcb=mcb, _zr=z_b1):
            zN, _, _ = _run_twoblock(pipe, prompt, seed, args,
                                     cache1=range(_a + 1, _b), cache2=[],
                                     freeze1=_a, freeze2=_b, m_clean_b=_mcb,
                                     velocity_inject={k: d})
            return zN.float() - _zr.float()

        _accumulate_and_save(prompt_dir, "twoblock", tag, (a, b, c), N,
                             defects, ("d_nonmem", "d_mem"), e_real, propagate)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 6: propagation closure test.")
    p.add_argument("--mode", choices=["oneblock", "twoblock"], required=True)
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--intervals", default="",
                   help="oneblock: comma 'a:b' intervals.")
    p.add_argument("--configs", default="",
                   help="twoblock: comma 'a:b:c' configs.")
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


def main() -> int:
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    N = int(args.num_steps)

    if args.mode == "oneblock":
        if not args.intervals:
            raise SystemExit("oneblock needs --intervals")
        units = [tuple(int(x) for x in t.split(":"))
                 for t in args.intervals.split(",") if t.strip()]
    else:
        if not args.configs:
            raise SystemExit("twoblock needs --configs")
        units = [tuple(int(x) for x in t.split(":"))
                 for t in args.configs.split(",") if t.strip()]

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
          f"mode={args.mode}; shard {args.shard_idx}/{args.shard_count} = "
          f"{len(shard_prompts)} prompts; {len(units)} units", flush=True)

    per_prompt = []
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

        if args.mode == "oneblock":
            _oneblock_prompt(pipe, prompt, seed, args, units, prompt_dir, N)
        else:
            _twoblock_prompt(pipe, prompt, seed, args, units, prompt_dir, N)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "closure_probe", "mode": args.mode,
            "prompt_idx": global_idx, "prompt": prompt, "seed": int(seed),
            "num_steps": N, "units": [list(u) for u in units],
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "closure_probe", "mode": args.mode, "num_steps": N,
        "shard_idx": int(args.shard_idx), "shard_count": int(args.shard_count),
        "base_seed": int(args.seed), "model_id": args.model_id,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
