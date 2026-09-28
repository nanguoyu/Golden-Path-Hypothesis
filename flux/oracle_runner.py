#!/usr/bin/env python3
"""Oracle fragility sweep on FLUX (R_k^oracle measurement).

For each prompt in the shard, runs:
  - 1 baseline trajectory (full computation at every step)
  - N-1 oracle trajectories where step k uses the chosen cache method's
    predicted residual (h_out = h_in + predicted_residual_k), all other
    steps are full.

Per docs/my_research_plan.md §10 Exp 1. The oracle quantifies how much the
final latent z_N drifts when only one specific step k is cached. Skipping
k=0 because no residual history is available before step 1.

`--cache_mode` selects which method's predictor defines the cache event.
The R_k^oracle measurement is method-specific: a SeaCache oracle measures
the harm of SeaCache-style caching at step k; a HiCache oracle measures
HiCache-style caching, etc. For per-method H3 verification, run one oracle
sweep per method and pair each method's Score_n against its own oracle in
analysis/score_compare.py.

Cache predictor at step k uses residual history from the most recent full
step (= k-1 in single-step setup), with step_offset = 1.
  - seacache:   pred = r_{k-1}                      (zero-order)
  - hicache:    pred = hicache_predict(history, ...) (Hermite O=2 sigma=0.5)
  - taylorseer: pred = taylor_predict(history, ...)  (Taylor O=1)

Output dir layout per shard:
    output_dir/
      prompt_XXXXX/
        baseline.pt           z_N latent (bf16, on CPU)
        baseline.png          decoded image
        oracle_k001.pt        z_N from "cache at step 1, full elsewhere"
        oracle_k001.png
        ...
        oracle_k049.pt
        oracle_k049.png
        manifest.json         prompt text, seed, file list, complete flag
      timing_shard{i}of{N}.json   per-shard aggregate timing

Cost per prompt (N=50): 50 trajectories × 50 forward = 2500 forwards.
On H100 bf16 ~500s/prompt. 4-GPU sharded 100 prompts ≈ 3.5h.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

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

from lib.hermite import hermite_update, hicache_predict  # noqa: E402
from lib.io_utils import read_prompts, split_shard  # noqa: E402
from lib.taylor import taylor_predict  # noqa: E402

CACHE_MODES = ("seacache", "hicache", "taylorseer")

logger = logging.get_logger(__name__)


# ----------------------------------------------------------------------------
# Oracle forward: identical structure to flux/seacache.py, simplified gating.
# Cache iff cnt == oracle_step_k AND previous_residual is not None.
# ----------------------------------------------------------------------------
def _oracle_forward(
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
    """Replacement for FluxTransformer2DModel.forward with deterministic gating."""
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

    # ---- Oracle gating: deterministic cache at scheduled steps --------------
    # Cache fires iff cur_step ∈ cache_steps AND we have any residual history
    # to base a prediction on. Cached steps do NOT update history (since we
    # never computed the true residual). For single-step oracle (Phase 1)
    # cache_steps is {oracle_step_k}; for multi-step (H4) it can be any subset.
    cur_step = -1
    should_calc = True
    if getattr(self, "enable_oracle", False):
        cur_step = int(self.cnt)
        if cur_step in self.cache_steps_set and self.residual_history:
            should_calc = False
        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0  # ready for next trajectory

    # ---- Block compute / skip ----------------------------------------------
    if (
        getattr(self, "enable_oracle", False)
        and not should_calc
        and self.residual_history
    ):
        # Method-dispatched cache prediction. step_offset is the integer
        # step distance from the last activation step to the current
        # (cache) step — = 1 in single-step oracle since the previous step
        # was always a full forward.
        step_offset = cur_step - int(self.last_activation_step)
        if self.cache_mode == "seacache":
            predicted_residual = self.residual_history[0]
        elif self.cache_mode == "hicache":
            predicted_residual = hicache_predict(
                self.residual_history,
                step_offset=int(step_offset),
                sigma=float(self.hicache_sigma),
                max_order=int(self.hicache_max_order),
            )
        elif self.cache_mode == "taylorseer":
            predicted_residual = taylor_predict(
                self.residual_history,
                step_offset=int(step_offset),
                max_order=int(self.taylorseer_max_order),
            )
        else:
            raise ValueError(f"unknown cache_mode: {self.cache_mode!r}")
        hidden_states = hidden_states + predicted_residual
    else:
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

        if getattr(self, "enable_oracle", False):
            actual_residual = hidden_states - ori_hidden_states
            # step_gap = distance from previous activation; first full call
            # has no previous activation → hermite_update returns {0: r}
            # regardless of step_gap (it short-circuits on empty prev).
            if self.last_activation_step < 0:
                step_gap = 1
            else:
                step_gap = cur_step - int(self.last_activation_step)
            # NOTE: in single-step oracle the cache only fires once per
            # trajectory; the post-cache step has step_gap=2 here and
            # subsequent steps revert to step_gap=1. hermite_update assumes
            # uniform-grid divided differences, so the Δ^{k>=1} entries
            # after the cache event are technically a mash-up of two grids.
            # This is benign because the cache branch only reads history
            # ONCE per trajectory (at oracle_step_k), where history was
            # last touched at oracle_step_k - 1 (uniform step_gap=1 up to
            # there). The post-cache update is dead state for the current
            # trajectory. Future extension to multi-step oracle would need
            # proper non-uniform divided differences.
            self.residual_history = hermite_update(
                self.residual_history,
                actual_residual.detach(),
                step_gap=int(step_gap),
                max_order=int(self.history_max_order),
            )
            self.last_activation_step = cur_step

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----------------------------------------------------------------------------
# install / teardown / reset
# ----------------------------------------------------------------------------
def install_oracle(
    pipe,
    *,
    oracle_step_k: Optional[int] = None,
    cache_steps: Optional[Any] = None,
    num_steps: int,
    cache_mode: str = "seacache",
    hicache_max_order: int = 2,
    hicache_sigma: float = 0.5,
    taylorseer_max_order: int = 1,
) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward for one oracle run.

    Exactly one of `oracle_step_k` (single-step Phase 1) or `cache_steps`
    (multi-step H4 / arbitrary subset) must be provided.

    Args:
        oracle_step_k: which single step (in 0..num_steps-1) to cache. k=0
            forced full because residual_history is empty before step 1.
            Convenience alias — internally stored as cache_steps_set = {k}.
        cache_steps: iterable of step indices (ints) to cache. Use this for
            multi-step oracle (H4 additivity test). k=0 still forced full
            via residual-history-emptiness guard.
        num_steps: total sampling steps.
        cache_mode: which method's predictor defines the cache event:
            "seacache" (zero-order reuse of previous residual),
            "hicache" (Hermite extrapolation, paper Table 1 settings),
            "taylorseer" (Taylor extrapolation, project default O=1).
        hicache_max_order, hicache_sigma: Hermite truncation order and dual-
            scaling factor (paper Table 1: O=2, sigma=0.5).
        taylorseer_max_order: Taylor truncation order (project default 1
            per CLAUDE.md — SeaCache paper FLUX convention).

    Note on history depth: `hermite_update` builds the residual history up to
    `max(hicache_max_order, taylorseer_max_order)` so the same dict can serve
    both predictors without silent truncation regardless of cache_mode. E.g.
    `--cache_mode taylorseer --taylorseer_max_order 2` will still get the
    Delta^2 entry it needs even if hicache_max_order=1.
    """
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {cache_mode!r}")
    if (oracle_step_k is None) == (cache_steps is None):
        raise ValueError(
            "install_oracle: exactly one of oracle_step_k or cache_steps must be set"
        )
    if oracle_step_k is not None:
        cache_steps_set = frozenset({int(oracle_step_k)})
    else:
        cache_steps_set = frozenset(int(k) for k in cache_steps)  # type: ignore[arg-type]

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _oracle_forward

    tr = pipe.transformer
    tr.enable_oracle = True
    tr.cache_steps_set = cache_steps_set
    # Kept for backward-compat / introspection: single-step case sets oracle_step_k.
    tr.oracle_step_k = (int(oracle_step_k) if oracle_step_k is not None
                        else (next(iter(cache_steps_set)) if len(cache_steps_set) == 1 else -1))
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr.cache_mode = str(cache_mode)
    tr.hicache_max_order = int(hicache_max_order)
    tr.hicache_sigma = float(hicache_sigma)
    tr.taylorseer_max_order = int(taylorseer_max_order)
    tr.history_max_order = max(int(hicache_max_order), int(taylorseer_max_order))
    tr.residual_history = {}
    tr.last_activation_step = -1

    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_oracle", "oracle_step_k", "cache_steps_set",
            "num_steps", "cnt",
            "cache_mode", "hicache_max_order", "hicache_sigma",
            "taylorseer_max_order", "history_max_order",
            "residual_history", "last_activation_step",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_oracle_state(pipe) -> None:
    """Reset per-trajectory state before each new pipe(...) call."""
    tr = pipe.transformer
    if hasattr(tr, "cnt"):
        tr.cnt = 0
    if hasattr(tr, "residual_history"):
        tr.residual_history = {}
    if hasattr(tr, "last_activation_step"):
        tr.last_activation_step = -1


# ----------------------------------------------------------------------------
# Pipe call + decode helpers
# ----------------------------------------------------------------------------
def _run_one_pipe_call(pipe, prompt: str, seed: int, args, callback=None) -> torch.Tensor:
    """One pipe(...) call returning the packed latent z_N (not decoded).

    `callback` is the pipeline's own `callback_on_step_end`; it is passed only
    when a caller asks for it, so the default call signature — and therefore
    every existing run — is unchanged. The hook reads `latents` and returns
    `callback_kwargs` unmodified (`lib/retained_trajectory.py`).
    """
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    extra = {}
    if callback is not None:
        extra = {
            "callback_on_step_end": callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
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
        **extra,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result.images  # tensor when output_type='latent'


def _decode_to_pil(pipe, latents: torch.Tensor, height: int, width: int):
    """Unpack + scale + VAE decode + postprocess to PIL Image. Mirrors
    diffusers FluxPipeline output_type='pil' path.

    `torch.no_grad()` is required: VAE decode otherwise produces a tensor with
    requires_grad=True, and image_processor.postprocess calls .numpy() which
    refuses grad tensors.
    """
    with torch.no_grad():
        latents_unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents_scaled = (latents_unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents_scaled, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


def _save_latent(z: torch.Tensor, path: Path) -> None:
    """Save z_N as bf16 on CPU."""
    torch.save(z.detach().to("cpu", dtype=torch.bfloat16), path)


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle fragility sweep on FLUX (R_k^oracle = ‖z_N^{cache-only-k} − z_N^full‖)."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after read, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose manifest.json marks complete.")
    p.add_argument("--cache_mode", choices=list(CACHE_MODES), default="seacache",
                   help="Which method's predictor defines the cache event at "
                        "step k. 'seacache' = zero-order reuse (default, "
                        "backward-compatible with pre-extension data).")
    p.add_argument("--hicache_max_order", type=int, default=2,
                   help="HiCache Hermite truncation order O (paper Tab 1: 2). "
                        "Also governs residual-history depth: history stores "
                        "Δ^0..Δ^O so taylorseer can downsample from the same.")
    p.add_argument("--hicache_sigma", type=float, default=0.5,
                   help="HiCache dual-scaling factor (paper Tab 1: 0.5).")
    p.add_argument("--taylorseer_max_order", type=int, default=1,
                   help="TaylorSeer expansion order (project default: 1, per "
                        "CLAUDE.md SeaCache-paper FLUX convention).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in {model_load_end - process_start:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} has {len(shard_prompts)} prompts "
          f"(global idx {start}..{end - 1})", flush=True)

    N = int(args.num_steps)
    k_values = list(range(1, N))   # skip k=0 (no residual history yet)
    H = (args.height // 16) * 16
    W = (args.width // 16) * 16

    print(f"[shard {args.shard_idx}] N={N}, sweep k in 1..{N-1} "
          f"({len(k_values)} oracle runs per prompt + 1 baseline), "
          f"cache_mode={args.cache_mode}", flush=True)

    per_prompt_records = []

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
        manifest_path = prompt_dir / "manifest.json"

        if args.resume and manifest_path.is_file():
            try:
                m = json.loads(manifest_path.read_text())
                if m.get("complete", False):
                    print(f"[shard {args.shard_idx}] prompt {global_idx} already complete, skip",
                          flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        prompt_dir.mkdir(parents=True, exist_ok=True)
        per_image_seed = args.seed + global_idx
        t_prompt = time.perf_counter()

        # ---- Baseline: no oracle install ------------------------------------
        z_base = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
        _save_latent(z_base, prompt_dir / "baseline.pt")
        _decode_to_pil(pipe, z_base, H, W).save(prompt_dir / "baseline.png")

        # ---- Oracle sweep ---------------------------------------------------
        for k in k_values:
            teardown = install_oracle(
                pipe,
                oracle_step_k=k,
                num_steps=N,
                cache_mode=str(args.cache_mode),
                hicache_max_order=int(args.hicache_max_order),
                hicache_sigma=float(args.hicache_sigma),
                taylorseer_max_order=int(args.taylorseer_max_order),
            )
            try:
                reset_oracle_state(pipe)
                z_k = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
            finally:
                teardown()
            _save_latent(z_k, prompt_dir / f"oracle_k{k:03d}.pt")
            _decode_to_pil(pipe, z_k, H, W).save(prompt_dir / f"oracle_k{k:03d}.png")

        t_done = time.perf_counter()
        prompt_seconds = float(t_done - t_prompt)

        manifest_data = {
            "prompt_idx": global_idx,
            "prompt": prompt,
            "seed": int(per_image_seed),
            "num_steps": N,
            "k_values": k_values,
            "files": {
                "baseline_latent": "baseline.pt",
                "baseline_image": "baseline.png",
                "oracle_latents": [f"oracle_k{k:03d}.pt" for k in k_values],
                "oracle_images": [f"oracle_k{k:03d}.png" for k in k_values],
            },
            "model_id": args.model_id,
            "model_name": args.model_name,
            "dtype": args.dtype,
            "guidance": float(args.guidance),
            "width": W,
            "height": H,
            "cache_mode": str(args.cache_mode),
            "hicache_max_order": int(args.hicache_max_order),
            "hicache_sigma": float(args.hicache_sigma),
            "taylorseer_max_order": int(args.taylorseer_max_order),
            "wall_seconds": prompt_seconds,
            "complete": True,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False))

        per_prompt_records.append({
            "global_idx": global_idx,
            "wall_seconds": prompt_seconds,
            "n_oracle_runs": len(k_values),
        })
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {prompt_seconds:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing_data = {
        "experiment": "oracle_fragility",
        "cache_mode": str(args.cache_mode),
        "cache_event": f"{args.cache_mode}_reuse",
        "hicache_max_order": int(args.hicache_max_order),
        "hicache_sigma": float(args.hicache_sigma),
        "taylorseer_max_order": int(args.taylorseer_max_order),
        "num_steps": N,
        "k_values_per_prompt": len(k_values),
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "base_seed": int(args.seed),
        "model_id": args.model_id,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "device": device_str,
        "model_load_s": float(model_load_end - process_start),
        "wallclock_total_s": float(process_end - process_start),
        "n_prompts_in_shard": len(shard_prompts),
        "per_prompt": per_prompt_records,
    }
    timing_path.write_text(json.dumps(timing_data, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}  "
          f"(total wall {process_end - process_start:.1f}s)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
