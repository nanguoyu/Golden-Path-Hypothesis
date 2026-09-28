#!/usr/bin/env python3
"""Experiment 0 instrumentation + Experiment 2 data generation.

docs/research_plan_after_extension.md §3. First version: `interval` mode,
zero-order residual reuse (seacache) only — per §0 修正6.

For an interval [a,b] (full at a, cache a+1..b-1, full from b) it produces
the quantities needed for Experiment 2 (one-block clean defect vs real
defect, with frozen memory so mu_k = 0):

  reference pass  (full trajectory):
    per step k:  z_k, v_full(z_k) = v_theta(z_k),
                 v_cache(z_k, m_a*) = C_k(z_k, m_a*)   for k > a
  cached pass     (full at every step in compute, cache a+1..b-1 to drive):
    per step k:  z~_k, v_full(z~_k) = v_theta(z~_k),
                 v_cache(z~_k, m_a*) = C_k(z~_k, m_a*) for k > a

where m_a* is the coarse residual frozen at the last full step a. The
instrumented forward always runs the full transformer blocks (so v_full is
available even on cached steps); the cache velocity is the cheap extra
G = norm_out + proj_out applied to (x_embed(z) + m_a*).

Downstream (analysis/exp2_state_feedback.py):
  d_clean_k = v_cache(z_k,  m_a*) - v_full(z_k)
  d_real_k  = v_cache(z~_k, m_a*) - v_full(z~_k)
  d_state_k = d_real_k - d_clean_k        (state-feedback + nonlinearity;
                                           mu_k = 0 so no memory-feedback)

Output dir layout per shard:
  output_dir/prompt_XXXXX/
    ref_aAA.pt              reference pass, one per distinct interval start a
    interval_aAA_bBB.pt     cached pass, one per interval
    manifest.json
Each .pt holds {"dump": [per-step dict], ...}; a per-step dict is
{"step", "is_cache", "z", "v_full", "v_cache"} (tensors bf16 on CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.io_utils import read_prompts, split_shard  # noqa: E402
from flux.oracle_runner import _run_one_pipe_call  # noqa: E402

logger = logging.get_logger(__name__)


# ----------------------------------------------------------------------------
# Instrumented forward: always runs full blocks; records v_full and (for
# k > freeze_step) v_cache = G(x_embed(z) + m_frozen). Drives the trajectory
# with the cache velocity at steps in cache_steps_set, else with v_full.
# ----------------------------------------------------------------------------
def _instr_forward(
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
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale") is not None:
        logger.warning(
            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        )

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

    cur_step = int(self.cnt)

    # ---- full block compute (always) ---------------------------------------
    ori_hidden_states = hidden_states
    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if controlnet_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
            )
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    for index_block, block in enumerate(self.single_transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if controlnet_single_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples))
            )
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

    actual_residual = hidden_states - ori_hidden_states

    # full velocity v_theta(z) = G(x_embed(z) + actual_residual)
    out_full = self.proj_out(self.norm_out(hidden_states, temb))

    # freeze the zero-order memory m_a* at the interval's last full step a
    if cur_step == int(self.freeze_step):
        self.m_frozen = actual_residual.detach()

    # cache velocity C_k(z, m_a*) = G(x_embed(z) + m_a*), for k > a
    out_cache = None
    if self.m_frozen is not None and cur_step > int(self.freeze_step):
        hc = ori_hidden_states + self.m_frozen
        out_cache = self.proj_out(self.norm_out(hc, temb))

    is_cache = (cur_step in self.cache_steps_set) and (out_cache is not None)
    returned = out_cache if is_cache else out_full

    self.dump.append({
        "step": cur_step,
        "is_cache": bool(is_cache),
        "v_full": out_full.detach().to("cpu", torch.bfloat16),
        "v_cache": (out_cache.detach().to("cpu", torch.bfloat16)
                    if out_cache is not None else None),
    })

    # Optional velocity injection (Exp 6 closure): add a per-step velocity
    # defect to the driving velocity, so the solver applies the drift
    # H_k * defect in fp32. Default {} -> no-op (Exp 2 behaviour unchanged).
    vinj = getattr(self, "velocity_inject", None)
    if vinj and cur_step in vinj:
        returned = returned + vinj[cur_step].to(returned)

    self.cnt += 1
    if self.cnt == int(self.num_steps):
        self.cnt = 0

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (returned,)
    return Transformer2DModelOutput(sample=returned)


def install_instr(pipe, *, cache_steps, freeze_step: int,
                  num_steps: int, velocity_inject=None) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward for one instrumented run.

    `velocity_inject` is an optional {step: defect} added to the driving
    velocity at that step (Exp 6 closure probe). Default None -> no-op.
    """
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _instr_forward
    tr = pipe.transformer
    tr.cache_steps_set = frozenset(int(k) for k in cache_steps)
    tr.freeze_step = int(freeze_step)
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr.m_frozen = None
    tr.dump = []
    tr.velocity_inject = ({int(k): v for k, v in velocity_inject.items()}
                          if velocity_inject else {})
    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("cache_steps_set", "freeze_step", "num_steps", "cnt",
                     "m_frozen", "dump", "velocity_inject"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def _select_ref(dump: List[Dict]) -> List[Dict]:
    """Reference records worth saving: steps k > a (v_cache available)."""
    return [{"step": r["step"], "v_full": r["v_full"], "v_cache": r["v_cache"]}
            for r in dump if r["v_cache"] is not None]


def _select_cache(dump: List[Dict]) -> List[Dict]:
    """Cached records worth saving: the cached steps a+1..b-1."""
    return [{"step": r["step"], "v_full": r["v_full"], "v_cache": r["v_cache"]}
            for r in dump if r["is_cache"]]


# ----------------------------------------------------------------------------
# Interval parsing
# ----------------------------------------------------------------------------
def parse_intervals(spec: str, num_steps: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        a_s, b_s = tok.split(":", 1)
        a, b = int(a_s), int(b_s)
        if not (0 <= a and a + 1 < b <= int(num_steps)):
            raise ValueError(f"interval [{a},{b}] invalid for N={num_steps}")
        out.append((a, b))
    if not out:
        raise ValueError("no intervals parsed")
    return sorted(set(out))


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Instrumented cache runner (Exp 0) — interval mode (Exp 2)."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--intervals", required=True,
                   help="comma-separated 'a:b' intervals.")
    p.add_argument("--mode", choices=["interval"], default="interval",
                   help="only 'interval' (Exp 2) in this version.")
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
    intervals = parse_intervals(args.intervals, N)
    by_a: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (a, b) in intervals:
        by_a[a].append((a, b))

    prompts_all = read_prompts(args.prompt_file,
                               limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}] empty slice, exiting.", flush=True)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts "
          f"(global {start}..{end - 1}); {len(intervals)} intervals, "
          f"{len(by_a)} distinct anchors", flush=True)

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

        for a, ivs in sorted(by_a.items()):
            # reference pass: full everywhere, m_a* frozen at a, v_cache for k>a
            teardown = install_instr(pipe, cache_steps=[], freeze_step=a, num_steps=N)
            try:
                pipe.transformer.cnt = 0
                pipe.transformer.dump = []
                pipe.transformer.m_frozen = None
                _run_one_pipe_call(pipe, prompt, seed, args)
                ref_dump = _select_ref(pipe.transformer.dump)
            finally:
                teardown()
            torch.save({"a": a, "num_steps": N, "kind": "reference",
                        "dump": ref_dump}, prompt_dir / f"ref_a{a:02d}.pt")

            # cached pass per interval with this anchor
            for (a2, b) in ivs:
                cache_steps = list(range(a + 1, b))
                teardown = install_instr(pipe, cache_steps=cache_steps,
                                         freeze_step=a, num_steps=N)
                try:
                    pipe.transformer.cnt = 0
                    pipe.transformer.dump = []
                    pipe.transformer.m_frozen = None
                    _run_one_pipe_call(pipe, prompt, seed, args)
                    cache_dump = _select_cache(pipe.transformer.dump)
                finally:
                    teardown()
                torch.save({"a": a, "b": b, "num_steps": N, "kind": "cached",
                            "cache_steps": cache_steps, "dump": cache_dump},
                           prompt_dir / f"interval_a{a:02d}_b{b:02d}.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "instrumented_exp2",
            "prompt_idx": global_idx, "prompt": prompt, "seed": int(seed),
            "num_steps": N, "intervals": [[a, b] for (a, b) in intervals],
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "instrumented_exp2", "num_steps": N,
        "shard_idx": int(args.shard_idx), "shard_count": int(args.shard_count),
        "base_seed": int(args.seed), "model_id": args.model_id,
        "dtype": args.dtype,
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path} "
          f"(total wall {process_end - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
