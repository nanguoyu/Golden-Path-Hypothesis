#!/usr/bin/env python3
"""Experiment SM-A: Level-2 predictive decomposition.

docs/research_plan_stateful_marginal.md §5 SM-A. Measures the per-cached-step
velocity-defect decomposition (plan §2.3)

    d^real = d^0 + d^state + d^mem

across multi-refresh cache policies, paired with the online features, to
test whether the features predict each term.

Definitions (zero-order residual reuse; c_n(z, M) = G(x_embed(z) + M),
G = norm_out . proj_out; f_n = full velocity; a_n = last refresh):

    d^0     = c_n(z_n , M_SL)  - f_n(z_n)     clean defect
    d^real  = c_n(z~_n, M_n )  - f_n(z~_n)    real defect
    d^state = c_n(z~_n, M_SL)  - f_n(z~_n) - d^0
    d^mem   = c_n(z~_n, M_n )  - c_n(z~_n, M_SL)

M_SL is the schedule-locked clean memory: the clean full residual at a_n,
built by re-running the SAME refresh schedule on the clean trajectory
(plan §2.1). M_n is the real memory from the drifted trajectory.

Per (prompt, policy) two instrumented runs (both always run the full block
stack): a clean run yields f(z_n), M_SL and c(z_n, M_SL); a policy run
yields f(z~_n), c(z~_n, M_n), c(z~_n, M_SL) and the gate-feature drift. The
four defects, their norms and the online features are assembled per cached
step.

Policies: interval (deterministic schedule) and the adaptive zero-order
methods SeaCache / TeaCache, whose realized schedule is resolved by a
native pre-run (forward hooks derive which steps were cached without
editing the locked cache-mode files). Higher-order predictors (HiCache /
TaylorSeer) need a multi-anchor memory-warmup definition and are SM-E.

Output per prompt: prompt_XXXXX/decomp_rows.json
  [{prompt_idx, policy, step, gap, q, p_acc, c_stale, c_traj, c_mem,
    psi_drift, norm_d0, norm_dreal, norm_dstate, norm_dmem,
    frac_state, frac_mem}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from lib.wiener import apply_sea_with_scheduler  # noqa: E402
from lib.marginal_features import MarginalFeatureTracker  # noqa: E402
from flux.oracle_runner import _run_one_pipe_call  # noqa: E402
from flux.seacache import install as seacache_install  # noqa: E402
from flux.teacache import install as teacache_install  # noqa: E402

logger = logging.get_logger(__name__)


def _interval_schedule(k: int, num_steps: int, first_enhance: int = 1
                       ) -> Tuple[frozenset, frozenset]:
    """(cache_set, refresh_set) for an interval-k policy.

    Full at step 0, the last step, the first `first_enhance` steps, and
    every k-th step; cache the rest — matching marginal_risk_probe's
    interval rule so the realized schedule is consistent across SM runners.
    """
    cache = set()
    for n in range(num_steps):
        force_full = (n == 0 or n == num_steps - 1 or n < first_enhance)
        if not force_full and n % k != 0:
            cache.add(n)
    refresh = frozenset(n for n in range(num_steps) if n not in cache)
    return frozenset(cache), refresh


# ---------------------------------------------------------------------------
# Instrumented forward: always runs the full block stack; mode "clean"
# builds the schedule-locked memory, mode "policy" drives the cached
# trajectory and evaluates the cache velocity at both memories.
# ---------------------------------------------------------------------------
def _decomp_forward(
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

    cur_step = int(self.dp_cnt)
    N = int(self.dp_num_steps)
    is_cached = cur_step in self.dp_cache_set
    ori_hidden_states = hidden_states            # = x_embed(z)

    # ---- gate feature psi (policy mode only, for the online features) -----
    psi_drift = 0.0
    if self.dp_mode == "policy":
        modulated_inp, *_ = self.transformer_blocks[0].norm1(ori_hidden_states, emb=temb)
        psi = modulated_inp.reshape(
            modulated_inp.shape[0],
            int(img_ids[:, 1].max().item() + 1),
            int(img_ids[:, 2].max().item() + 1),
            modulated_inp.shape[-1],
        )
        psi = apply_sea_with_scheduler(psi, self.scheduler, cur_step,
                                       power_exp=2.0, dims=(-2, -3), norm_mode="mean")
        psi = psi.reshape(psi.shape[0], -1, psi.shape[-1])
        if self.dp_prev_psi is not None:
            psi_drift = rel_l1(psi, self.dp_prev_psi)
        self.dp_prev_psi = psi.detach()

    # ---- full block stack (always) ---------------------------------------
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

    actual_residual = hidden_states - ori_hidden_states

    def _G(h):
        return self.proj_out(self.norm_out(h, temb))

    out_full = _G(hidden_states)                 # f_n(z)

    rec: Dict[str, Any] = {"step": cur_step, "is_cached": bool(is_cached)}

    if self.dp_mode == "clean":
        if not is_cached:                        # refresh: update M_SL
            self.dp_m_sl = actual_residual.detach()
            self.dp_m_sl_by_refresh[cur_step] = self.dp_m_sl
        rec["v_full"] = out_full.detach().to("cpu", torch.float32)
        if is_cached and self.dp_m_sl is not None:
            rec["v_cache_sl"] = _G(ori_hidden_states + self.dp_m_sl).detach().to("cpu", torch.float32)
        returned = out_full                      # clean run drives full
    else:                                        # policy mode
        rec["u"] = 0 if not is_cached else 1
        rec["psi_drift"] = float(psi_drift)
        if not is_cached:                        # refresh: update M_n
            self.dp_m_real = actual_residual.detach()
            self.dp_last_refresh = cur_step
            returned = out_full
        else:
            m_sl = self.dp_m_sl_inject.get(self.dp_last_refresh)
            v_cache_real = _G(ori_hidden_states + self.dp_m_real)
            rec["v_full"] = out_full.detach().to("cpu", torch.float32)
            rec["v_cache_real"] = v_cache_real.detach().to("cpu", torch.float32)
            if m_sl is not None:
                m_sl = m_sl.to(ori_hidden_states.device, ori_hidden_states.dtype)
                rec["v_cache_sl"] = _G(ori_hidden_states + m_sl).detach().to("cpu", torch.float32)
            returned = v_cache_real              # policy drives with cache
    self.dp_rec.append(rec)

    self.dp_cnt += 1
    if self.dp_cnt == N:
        self.dp_cnt = 0

    output = returned
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install_decomp(pipe, *, mode: str, cache_set, num_steps: int,
                   m_sl_inject: Optional[Dict[int, torch.Tensor]] = None):
    """Patch FluxTransformer2DModel.forward with the SM-A decomposition probe."""
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _decomp_forward
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.dp_mode = str(mode)
    tr.dp_cache_set = frozenset(int(k) for k in cache_set)
    tr.dp_num_steps = int(num_steps)
    tr.dp_cnt = 0
    tr.dp_rec = []
    tr.dp_m_sl = None
    tr.dp_m_sl_by_refresh = {}
    tr.dp_m_real = None
    tr.dp_last_refresh = 0
    tr.dp_prev_psi = None
    tr.dp_m_sl_inject = dict(m_sl_inject) if m_sl_inject else {}
    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("scheduler", "dp_mode", "dp_cache_set", "dp_num_steps",
                     "dp_cnt", "dp_rec", "dp_m_sl", "dp_m_sl_by_refresh",
                     "dp_m_real", "dp_last_refresh", "dp_prev_psi",
                     "dp_m_sl_inject"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def _run_decomp(pipe, prompt, seed, args, *, mode, cache_set, m_sl_inject=None
                ) -> Tuple[List[Dict], Dict[int, torch.Tensor]]:
    """One instrumented run; return (per-step records, M_SL-by-refresh)."""
    teardown = install_decomp(pipe, mode=mode, cache_set=cache_set,
                              num_steps=int(args.num_steps),
                              m_sl_inject=m_sl_inject)
    try:
        _run_one_pipe_call(pipe, prompt, seed, args)
        tr = pipe.transformer
        rec = sorted(tr.dp_rec, key=lambda r: r["step"])
        m_sl = dict(tr.dp_m_sl_by_refresh)
    finally:
        teardown()
    return rec, m_sl


def _decomp_rows(global_idx: int, pname: str, cache_set,
                 clean_rec: List[Dict], policy_rec: List[Dict],
                 q_abs, s_cal, a_cal, mem_variant: int) -> List[Dict]:
    """Assemble the four defects + online features into per-cached-step rows."""
    clean = {r["step"]: r for r in clean_rec}
    policy = {r["step"]: r for r in policy_rec}

    tracker = MarginalFeatureTracker(q_abs, s_cal, a_cal, mem_variant=mem_variant)
    feats: Dict[int, Dict[str, float]] = {}
    for r in policy_rec:
        feats[r["step"]] = dict(tracker.observe(r["step"], r.get("psi_drift", 0.0)))
        tracker.commit(r["step"], int(r["u"]))

    rows = []
    for k in sorted(cache_set):
        cr, pr = clean.get(k), policy.get(k)
        if cr is None or pr is None:
            continue
        if "v_cache_sl" not in cr or "v_cache_sl" not in pr:
            continue
        d0 = cr["v_cache_sl"] - cr["v_full"]
        d_real = pr["v_cache_real"] - pr["v_full"]
        d_state = (pr["v_cache_sl"] - pr["v_full"]) - d0
        d_mem = pr["v_cache_real"] - pr["v_cache_sl"]
        n_real = float(d_real.norm()) + 1e-12
        f = feats[k]
        rows.append({
            "prompt_idx": global_idx, "policy": pname, "step": k,
            "gap": f["gap"], "q": f["q"], "p_acc": f["p_acc"],
            "c_stale": f["c_stale"], "c_traj": f["c_traj"], "c_mem": f["c_mem"],
            "psi_drift": f["psi_drift"],
            "norm_d0": float(d0.norm()),
            "norm_dreal": float(d_real.norm()),
            "norm_dstate": float(d_state.norm()),
            "norm_dmem": float(d_mem.norm()),
            "frac_state": float(d_state.norm()) / n_real,
            "frac_mem": float(d_mem.norm()) / n_real,
        })
    return rows


def _resolve_adaptive_schedule(pipe, prompt, seed, args, kind: str,
                               param: float) -> frozenset:
    """Realized cached-step set of a native adaptive policy.

    Runs native SeaCache / TeaCache and derives which steps were cached via
    forward hooks: a forward-pre-hook on transformer_blocks[0] fires only
    when the block stack runs (a full step) — the gate's separate
    `transformer_blocks[0].norm1(...)` call does not trigger it — while
    proj_out fires every step. No edit to the locked cache-mode files.
    """
    N = int(args.num_steps)
    if kind == "seacache":
        teardown = seacache_install(pipe, threshold=float(param),
                                    num_steps=N, first_enhance=1)
    elif kind == "teacache":
        teardown = teacache_install(pipe, threshold=float(param),
                                    num_steps=N, first_enhance=1)
    else:
        raise ValueError(f"adaptive kind must be seacache/teacache, got {kind!r}")

    st = {"step": 0, "blocks_ran": False, "full": set()}
    tr = pipe.transformer

    def _pre_block(_m, _inp):
        st["blocks_ran"] = True

    def _post_proj(_m, _inp, _out):
        if st["blocks_ran"]:
            st["full"].add(st["step"])
        st["blocks_ran"] = False
        st["step"] += 1

    h_block = tr.transformer_blocks[0].register_forward_pre_hook(_pre_block)
    h_proj = tr.proj_out.register_forward_hook(_post_proj)
    try:
        _run_one_pipe_call(pipe, prompt, seed, args)
    finally:
        h_block.remove()
        h_proj.remove()
        teardown()
    return frozenset(n for n in range(N) if n not in st["full"])


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


def _parse_policies(spec: str, num_steps: int
                    ) -> List[Tuple[str, str, Optional[frozenset], Optional[float]]]:
    """[(name, kind, cache_set_or_None, param)].

    interval_* schedules are deterministic and resolved here; seacache_* /
    teacache_* are adaptive — cache_set is None and resolved per prompt.
    """
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("interval_"):
            k = int(tok.split("_")[1])
            cache, _ = _interval_schedule(k, num_steps)
            out.append((tok, "interval", cache, None))
        elif tok.startswith("seacache_"):
            out.append((tok, "seacache", None, float(tok.split("_", 1)[1])))
        elif tok.startswith("teacache_"):
            out.append((tok, "teacache", None, float(tok.split("_", 1)[1])))
        else:
            raise ValueError(f"unknown policy token: {tok!r}")
    if not out:
        raise ValueError("no policies parsed")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp SM-A: Level-2 predictive decomposition.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--q_k", type=Path, required=True)
    p.add_argument("--sa_calib", type=Path, required=True)
    p.add_argument("--policies",
                   default="interval_2,interval_4,interval_6,"
                           "seacache_0.3,seacache_0.6,teacache_0.4,teacache_0.6")
    p.add_argument("--mem_variant", type=int, choices=[0, 1, 2], default=1)
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
    policies = _parse_policies(args.policies, N)
    q_abs, s_cal, a_cal = _load_calib(args.q_k, args.sa_calib, N)

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
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts; "
          f"{len(policies)} policies", flush=True)

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

        rows: List[Dict] = []
        for (pname, kind, cache_set, param) in policies:
            if kind != "interval":
                cache_set = _resolve_adaptive_schedule(pipe, prompt, seed, args,
                                                       kind, param)
            clean_rec, m_sl = _run_decomp(pipe, prompt, seed, args,
                                          mode="clean", cache_set=cache_set)
            policy_rec, _ = _run_decomp(pipe, prompt, seed, args, mode="policy",
                                        cache_set=cache_set, m_sl_inject=m_sl)
            rows.extend(_decomp_rows(global_idx, pname, cache_set,
                                     clean_rec, policy_rec,
                                     q_abs, s_cal, a_cal, int(args.mem_variant)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (prompt_dir / "decomp_rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False))
        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "marginal_decomp_probe", "prompt_idx": global_idx,
            "prompt": prompt, "seed": int(seed), "num_steps": N,
            "policies": [p[0] for p in policies], "n_rows": len(rows),
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)}) rows={len(rows)}", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "marginal_decomp_probe", "num_steps": N,
        "policies": [p[0] for p in policies], "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count), "base_seed": int(args.seed),
        "model_id": args.model_id, "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
