#!/usr/bin/env python3
"""Experiment 3: two-block memory-feedback test.

docs/research_plan_after_extension.md §3 Experiment 3. Zero-order residual
reuse only (plan §0 修正6 — one full step cannot reset a higher-order
memory).

A two-block config [a,b,c]:
  full 0..a -> cache a+1..b-1 (block 1) -> full refresh at b
            -> cache b+1..c-1 (block 2) -> full c..N

Block 1 drifts the latent, so the full refresh at b builds the real memory
  m~_b = F(z~_b)   on the drifted latent,
versus the clean reference memory
  m*_b = F(z_b)    on the reference latent.
The memory error mu_b = m~_b - m*_b is what activates K^m mu — it is
identically zero in the one-block Experiment 2.

In block 2, on the SAME drifted latent z~_k, the runner records:
  v_full        = v_theta(z~_k)
  v_cache_real  = C_k(z~_k, m~_b) = G(x_embed(z~_k) + m~_b)
  v_cache_clean = C_k(z~_k, m*_b) = G(x_embed(z~_k) + m*_b)
=> d_real = v_cache_real - v_full,  d_state_only = v_cache_clean - v_full,
   d_mem  = d_real - d_state_only = v_cache_real - v_cache_clean.
d_mem is the memory-feedback term. The trajectory is driven by
v_cache_real (the real run uses the real memory).

m*_b is supplied by a reference (full) pass that records F(z_b).

Output per prompt: prompt_XXXXX/twoblock_aAA_bBB_cCC.pt
  {a,b,c, num_steps, norm_mu_b, dump:[{step,v_full,v_cache_real,v_cache_clean}]}
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


def _twoblock_forward(
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
    out_full = self.proj_out(self.norm_out(hidden_states, temb))

    if cur_step == int(self.freeze1):
        self.m1 = actual_residual.detach()
    if cur_step == int(self.freeze2):
        self.m2 = actual_residual.detach()
    if cur_step in self.record_residual_steps:
        self.residuals[cur_step] = actual_residual.detach()

    if cur_step in self.cache1_set and self.m1 is not None:
        returned = self.proj_out(self.norm_out(ori_hidden_states + self.m1, temb))
    elif cur_step in self.cache2_set and self.m2 is not None:
        out_real = self.proj_out(self.norm_out(ori_hidden_states + self.m2, temb))
        out_clean = self.proj_out(self.norm_out(
            ori_hidden_states + self.m_clean_b, temb))
        returned = out_real
        self.dump.append({
            "step": cur_step,
            "v_full": out_full.detach().to("cpu", torch.bfloat16),
            "v_cache_real": out_real.detach().to("cpu", torch.bfloat16),
            "v_cache_clean": out_clean.detach().to("cpu", torch.bfloat16),
        })
    else:
        returned = out_full

    # Optional velocity injection (Exp 6 closure): add a per-step velocity
    # defect to the driving velocity, so the solver applies the drift
    # H_k * defect in fp32. Default {} -> no-op (Exp 3 behaviour unchanged).
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


def install_twoblock(pipe, *, cache1, cache2, freeze1: int, freeze2: int,
                     m_clean_b: Optional[torch.Tensor],
                     record_residual_steps, num_steps: int,
                     velocity_inject=None) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward for one two-block run.

    `velocity_inject` is an optional {step: defect} added to the driving
    velocity at that step (Exp 6 closure probe). Default None -> no-op.
    """
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _twoblock_forward
    tr = pipe.transformer
    tr.cache1_set = frozenset(int(k) for k in cache1)
    tr.cache2_set = frozenset(int(k) for k in cache2)
    tr.freeze1 = int(freeze1)
    tr.freeze2 = int(freeze2)
    tr.m_clean_b = m_clean_b
    tr.record_residual_steps = frozenset(int(k) for k in record_residual_steps)
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr.m1 = None
    tr.m2 = None
    tr.residuals = {}
    tr.dump = []
    tr.velocity_inject = ({int(k): v for k, v in velocity_inject.items()}
                          if velocity_inject else {})
    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("cache1_set", "cache2_set", "freeze1", "freeze2",
                     "m_clean_b", "record_residual_steps", "num_steps", "cnt",
                     "m1", "m2", "residuals", "dump", "velocity_inject"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def parse_configs(spec: str, num_steps: int) -> List[Tuple[int, int, int]]:
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        a_s, b_s, c_s = tok.split(":")
        a, b, c = int(a_s), int(b_s), int(c_s)
        if not (0 <= a and a + 1 < b and b + 1 < c <= int(num_steps)):
            raise ValueError(f"two-block [{a},{b},{c}] invalid for N={num_steps}")
        out.append((a, b, c))
    if not out:
        raise ValueError("no two-block configs parsed")
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3: two-block memory-feedback.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--configs", required=True,
                   help="comma-separated 'a:b:c' two-block configs.")
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
    configs = parse_configs(args.configs, N)
    b_steps = sorted({b for (_, b, _) in configs})

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
          f"{len(configs)} two-block configs", flush=True)

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

        # reference pass: full everywhere, record F(z_b) = m*_b for each b
        teardown = install_twoblock(
            pipe, cache1=[], cache2=[], freeze1=0, freeze2=0,
            m_clean_b=None, record_residual_steps=b_steps, num_steps=N)
        try:
            _run_one_pipe_call(pipe, prompt, seed, args)
            m_clean = {b: pipe.transformer.residuals[b] for b in b_steps}
        finally:
            teardown()

        # one two-block run per config
        for (a, b, c) in configs:
            teardown = install_twoblock(
                pipe, cache1=range(a + 1, b), cache2=range(b + 1, c),
                freeze1=a, freeze2=b, m_clean_b=m_clean[b],
                record_residual_steps=[], num_steps=N)
            try:
                _run_one_pipe_call(pipe, prompt, seed, args)
                dump = list(pipe.transformer.dump)
                m2 = pipe.transformer.m2
                norm_mu_b = float((m2.float() - m_clean[b].float()).norm())
            finally:
                teardown()
            torch.save({"a": a, "b": b, "c": c, "num_steps": N,
                        "norm_mu_b": norm_mu_b, "dump": dump},
                       prompt_dir / f"twoblock_a{a:02d}_b{b:02d}_c{c:02d}.pt")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "twoblock_probe", "prompt_idx": global_idx,
            "prompt": prompt, "seed": int(seed), "num_steps": N,
            "configs": [[a, b, c] for (a, b, c) in configs],
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "twoblock_probe", "num_steps": N,
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
