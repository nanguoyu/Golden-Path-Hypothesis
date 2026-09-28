"""SQA E1 forced-stale-gap probe.

`docs/research_plan_method_native_sqa.md` §6 E1 defines a synthetic
stale-memory single-cache action `(a, n)` and the label

    Y_Sea(a, n) = ||z_N^{force cache at n using R_a, full after}
                   - z_N^{force full at n, full after}||

This script computes that label, together with the byte-faithful native
accumulator value `P_Sea(a, n)` (per plan §4.1, replayed via
`lib/sqa_replay.py`), for every prompt × action in the pinned action
grid.

Per-prompt flow (one `pipe(...)` call per phase):

    Phase A — trace
        Install `flux/sqa_trace.py:install_trace`, run a full no-cache
        FLUX trajectory, pop the captured trace, teardown. The trace
        holds per-step (psi_raw, psi_filtered, residual) on CPU bf16.
        z_N^full comes from the pipe return.

    Phase B — branch-A per action
        For each (a, n, g) in the action grid, install the inject hook
        with R_a = trace["residual"][a] (moved back to GPU), run a fresh
        `pipe(...)` with the same seed, then teardown. At step n the
        hook short-circuits the block stack with `h + R_a`, mirroring
        SeaCache's cache action at line 168 of flux/seacache.py. z_N^A
        comes from the pipe return.

        Y_Sea = ||z_N^A - z_N^full||_2  (fp32)
        P_Sea = replay_p_sea_from_trace(trace, a, n)

S_n, Q_n, A_n are NOT written into the raw rows — the downstream
analysis loads them from the sa_calib JSON + scheduler and computes
V_n = S Q A. This keeps the option to swap calibration tables (legacy
vs clean-RMS) without re-running the probe.

Action grid (plan §6 E1, lines 606-619 of the plan doc, total 40
actions per prompt for `--action_grid full`, 10 for `--action_grid smoke`).
The grid is hard-coded in this file rather than read from JSON because
it is a research-design constant. Changing it requires a code edit and a
git commit so the provenance is recorded.

Output:
    <output_dir>/prompt_<global_idx:05d>/sqa_e1_rows.json
    {
      "prompt_idx": int,
      "prompt":     str,
      "num_steps":  int,
      "seed":       int,
      "action_grid": "smoke" | "full",
      "rows": [
        {"a": int, "n": int, "gap": int,
         "Y_Sea": float, "P_Sea": float},
        ...
      ]
    }

Sharding follows the project convention (`flux/oracle_runner.py`):
one shard = one Slurm task. Pass --shard_idx / --shard_count.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sys
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from flux.sqa_trace import install_trace, pop_trace
from lib.io_utils import read_prompts, split_shard
from lib.sqa_replay import replay_p_sea_from_trace

logger = logging.get_logger(__name__)


# =============================================================================
# Action grid — pinned from docs/research_plan_method_native_sqa.md §6 E1
# =============================================================================

def _build_full_grid() -> List[Tuple[int, int, int]]:
    """40 actions, balanced across gap buckets, spanning early/mid/late
    non-final cacheable steps. Returns list of (a, n, gap)."""
    grid: List[Tuple[int, int, int]] = []
    # g in {1, 2, 4} with n in {4,10,16,22,28,34,40,48}
    for g in (1, 2, 4):
        for n in (4, 10, 16, 22, 28, 34, 40, 48):
            a = n - g
            grid.append((a, n, g))
    # g = 8 with n in {8,14,20,26,32,38,44,48}
    for n in (8, 14, 20, 26, 32, 38, 44, 48):
        g = 8
        a = n - g
        grid.append((a, n, g))
    # g = 16 with n in {16,20,24,28,32,36,40,48}
    for n in (16, 20, 24, 28, 32, 36, 40, 48):
        g = 16
        a = n - g
        grid.append((a, n, g))
    assert len(grid) == 40, f"action grid has {len(grid)} actions, expected 40"
    # Verify (a, n) uniqueness
    assert len({(a, n) for a, n, _ in grid}) == 40, "action grid has duplicate (a,n) pairs"
    return grid


def _build_smoke_grid() -> List[Tuple[int, int, int]]:
    """10-action smoke subset (plan §6 E1 lines 617-619)."""
    pairs = [
        (1, 16), (1, 40),
        (2, 16), (2, 40),
        (4, 16), (4, 40),
        (8, 20), (8, 44),
        (16, 24), (16, 48),
    ]
    grid = [(n - g, n, g) for (g, n) in pairs]
    assert len(grid) == 10
    return grid


ACTION_GRID_FULL = _build_full_grid()
ACTION_GRID_SMOKE = _build_smoke_grid()


# =============================================================================
# Inject forward — branch A: at step n, short-circuit block stack with R_a
# =============================================================================

def _inject_forward(
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
    """Drop-in replacement for `FluxTransformer2DModel.forward` that injects
    a logged residual at one specific step `n_inject`.

    Behavior:
      * step k != n_inject: run the full block stack (byte-equivalent to
        stock diffusers FluxTransformer2DModel.forward).
      * step k == n_inject: skip the block stack, return
        `hidden_states + self.injected_residual` (same as
        flux/seacache.py:168 cache-reuse path).

    Counter wraparound on `cnt == num_steps` matches the convention used in
    flux/seacache.py and flux/sqa_trace.py.
    """
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
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
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

    # ---- Inject gate -------------------------------------------------------
    should_inject = False
    if getattr(self, "enable_sqa_inject", False):
        cur_step = int(self.cnt)
        should_inject = (cur_step == int(self.n_inject))
        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0

    # ---- Block compute / inject ---------------------------------------------
    if should_inject:
        hidden_states = hidden_states + self.injected_residual
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                def _ckpt(module):
                    def _fwd(hs, ehs, temb_, ire):
                        return module(
                            hidden_states=hs,
                            encoder_hidden_states=ehs,
                            temb=temb_,
                            image_rotary_emb=ire,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return _fwd

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt(block), hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb, **ckpt_kwargs,
                )
            else:
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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                def _ckpt2(module):
                    def _fwd(hs, ehs, temb_, ire):
                        return module(
                            hidden_states=hs,
                            encoder_hidden_states=ehs,
                            temb=temb_,
                            image_rotary_emb=ire,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return _fwd

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt2(block), hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb, **ckpt_kwargs,
                )
            else:
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

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install_inject(
    pipe,
    *,
    num_steps: int,
    n_inject: int,
    residual_a: torch.Tensor,
) -> Callable[[], None]:
    """Patch `FluxTransformer2DModel.forward` to inject `residual_a` at step
    `n_inject`. Returns a teardown callable. Idempotent.

    Args:
        pipe: a loaded `DiffusionPipeline`.
        num_steps: total sampling steps (for end-of-trajectory counter wrap).
        n_inject: step index at which to substitute the block-stack output
            with `hidden_states + residual_a`.
        residual_a: pre-loaded whole-transformer residual to inject. Must
            already be on the same device + dtype as the transformer's hidden
            states (caller is responsible for the `.to(device, dtype)` move).

    Returns:
        teardown: zero-arg callable, restores the original forward and clears
        the per-instance state. Idempotent.
    """
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _inject_forward

    tr = pipe.transformer
    tr.enable_sqa_inject = True
    tr.num_steps = int(num_steps)
    tr.n_inject = int(n_inject)
    tr.injected_residual = residual_a
    tr.cnt = 0

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_sqa_inject", "num_steps", "n_inject",
            "injected_residual", "cnt",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


# =============================================================================
# Per-prompt orchestrator
# =============================================================================

def _run_one_pipe_call(pipe, prompt: str, seed: int, args) -> torch.Tensor:
    """One pipe(...) call returning the packed latent z_N."""
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        num_inference_steps=int(args.num_steps),
        guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result.images  # tensor when output_type='latent'


def _process_prompt(
    pipe,
    *,
    global_idx: int,
    prompt: str,
    actions: List[Tuple[int, int, int]],
    args,
    torch_dtype,
) -> Dict[str, Any]:
    """Run trace + all branches for one prompt. Returns the output dict that
    will be written to `prompt_NNNNN/sqa_e1_rows.json`."""
    seed = int(args.seed) + int(global_idx)
    device = pipe.device

    # ---- Phase A: trace ---------------------------------------------------
    t0 = time.perf_counter()
    teardown_trace = install_trace(pipe, num_steps=int(args.num_steps))
    try:
        z_N_full = _run_one_pipe_call(pipe, prompt, seed, args)
        trace = pop_trace(pipe)
    finally:
        teardown_trace()
    t_trace = time.perf_counter() - t0

    # Sanity: trace length matches num_steps
    n_recorded = len(trace["psi_raw"])
    if n_recorded != int(args.num_steps):
        raise RuntimeError(
            f"trace length mismatch: recorded {n_recorded} steps, expected "
            f"{args.num_steps}. (Re-check trace counter wraparound.)"
        )

    # Cache z_N_full on fp32 CPU for diff with later branches
    z_N_full_fp32 = z_N_full.detach().to(device).to(torch.float32)

    # ---- Phase B: branch-A per action ------------------------------------
    rows: List[Dict[str, Any]] = []
    t_branches_start = time.perf_counter()
    for action_idx, (a, n, gap) in enumerate(actions):
        # Replay P_Sea from trace (CPU bf16 tensors — rel_l1 promotes to fp32
        # internally for the abs/mean ops).
        p_sea = replay_p_sea_from_trace(trace, a=int(a), n=int(n))

        # Inject branch
        residual_a_gpu = trace["residual"][int(a)].to(device, dtype=torch_dtype)
        teardown_inject = install_inject(
            pipe,
            num_steps=int(args.num_steps),
            n_inject=int(n),
            residual_a=residual_a_gpu,
        )
        try:
            z_N_A = _run_one_pipe_call(pipe, prompt, seed, args)
        finally:
            teardown_inject()
        # Free GPU residual immediately
        del residual_a_gpu

        # Y_Sea = ||z_N^A - z_N^full|| in fp32
        diff = z_N_A.to(torch.float32) - z_N_full_fp32
        y_sea = float(diff.norm().item())
        del diff, z_N_A

        rows.append({
            "a": int(a),
            "n": int(n),
            "gap": int(gap),
            "Y_Sea": y_sea,
            "P_Sea": float(p_sea),
        })

    t_branches = time.perf_counter() - t_branches_start

    # Free trace
    del trace, z_N_full, z_N_full_fp32
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prompt_idx": int(global_idx),
        "prompt": prompt,
        "num_steps": int(args.num_steps),
        "seed": int(seed),
        "action_grid": args.action_grid,
        "n_actions": len(actions),
        "trace_seconds": float(t_trace),
        "branches_seconds": float(t_branches),
        "rows": rows,
    }


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SQA E1 forced-stale-gap probe (plan §6 E1) — SeaCache only."
    )
    p.add_argument("--prompt_file", type=Path, required=True,
                   help="Prompt list (one per line).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output dir for prompt_<idx>/sqa_e1_rows.json.")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; per-prompt seed = seed + global_idx, "
                        "matching the Phase-1 oracle convention.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--action_grid", choices=["smoke", "full"], default="smoke",
                   help="`smoke` = 10 pinned (a,n) actions; `full` = 40 actions. "
                        "See docs/research_plan_method_native_sqa.md §6 E1.")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--prompt_offset", type=int, default=0,
                   help="First global prompt index to use before applying --limit. "
                        "The output prompt_idx and seed remain in the original "
                        "prompt-file coordinate system.")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after --prompt_offset, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose sqa_e1_rows.json already exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if int(args.prompt_offset) < 0:
        raise SystemExit(f"--prompt_offset must be non-negative, got {args.prompt_offset}")
    prompts_full = read_prompts(args.prompt_file, limit=None)
    if int(args.prompt_offset) >= len(prompts_full):
        raise SystemExit(
            f"--prompt_offset={args.prompt_offset} is outside prompt file with "
            f"{len(prompts_full)} prompts"
        )
    slice_end = (
        len(prompts_full)
        if int(args.limit) <= 0
        else min(len(prompts_full), int(args.prompt_offset) + int(args.limit))
    )
    prompts_all = prompts_full[int(args.prompt_offset):slice_end]
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = ACTION_GRID_SMOKE if args.action_grid == "smoke" else ACTION_GRID_FULL

    # Validate action grid is reachable at the configured num_steps.
    max_n = max(n for _, n, _ in actions)
    if max_n >= int(args.num_steps):
        raise SystemExit(
            f"action grid requires n < num_steps; max n in {args.action_grid} "
            f"grid is {max_n}, --num_steps={args.num_steps}."
        )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} "
          f"dtype={args.dtype}", flush=True)
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in "
        f"{model_load_end - process_start:.1f}s; shard "
        f"{args.shard_idx}/{args.shard_count} has {len(shard_prompts)} prompts "
        f"(global idx {int(args.prompt_offset) + start}..{int(args.prompt_offset) + end - 1}); "
        f"action_grid={args.action_grid} "
        f"({len(actions)} actions/prompt)",
        flush=True,
    )

    skipped = 0
    completed = 0
    failed = 0

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = int(args.prompt_offset) + start + local_idx
        out_dir = args.output_dir / f"prompt_{global_idx:05d}"
        out_path = out_dir / "sqa_e1_rows.json"
        if args.resume and out_path.is_file():
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        t_p = time.perf_counter()
        try:
            result = _process_prompt(
                pipe,
                global_idx=global_idx,
                prompt=prompt,
                actions=actions,
                args=args,
                torch_dtype=torch_dtype,
            )
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            completed += 1
            t_pp = time.perf_counter() - t_p
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] prompt {global_idx} "
                f"DONE in {t_pp:.1f}s (trace={result['trace_seconds']:.1f}s, "
                f"branches={result['branches_seconds']:.1f}s, "
                f"{result['n_actions']} actions)",
                flush=True,
            )
        except Exception as e:
            failed += 1
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] prompt {global_idx} "
                f"FAILED ({type(e).__name__}: {e})",
                flush=True,
            )
            # Continue with next prompt; do not abort the shard

    total_s = time.perf_counter() - process_start
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] shard "
        f"{args.shard_idx}/{args.shard_count} done. "
        f"completed={completed} skipped={skipped} failed={failed} "
        f"total={total_s:.1f}s",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
