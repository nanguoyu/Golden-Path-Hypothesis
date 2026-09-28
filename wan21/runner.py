#!/usr/bin/env python3
"""Single-GPU Wan2.1 T2V runner with coarse cache/payload modes."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.io_utils import write_timing_json  # noqa: E402
from wan21._helpers import (  # noqa: E402
    atomic_write_json,
    build_manifest,
    decisions_filename,
    load_locked_schedule,
    read_prompt_shard,
    seed_for,
    video_filename,
)
from wan21.fine_payload import (  # noqa: E402
    FINE_PAYLOAD_MODES,
    FinePayloadConfig,
    decisions as fine_payload_decisions,
    install_fine_payload_forward,
    reset_fine_payload_state,
)
from wan21.payload import PAYLOAD_MODES as COARSE_PAYLOAD_MODES  # noqa: E402
from wan21.seacache import (  # noqa: E402
    CacheForwardConfig,
    decisions as cache_decisions,
    install_cache_forward,
    reset_cache_state,
    restore_original_forwards,
)
from wan21.taylorseer_fine import (  # noqa: E402
    TaylorSeerFineConfig,
    decisions as taylorseer_fine_decisions,
    install_hicache_fine_forward,
    install_taylorseer_fine_forward,
    reset_hicache_fine_state,
    reset_taylorseer_fine_state,
)
from wan21.segment_payload import (  # noqa: E402
    SEGMENT_LAYOUTS,
    SEGMENT_PAYLOAD_MODES,
    SegmentPayloadConfig,
    decisions as segment_payload_decisions,
    install_segment_payload_forward,
    reset_segment_payload_state,
)


LOG = logging.getLogger("wan21.runner")
PAYLOAD_MODES = COARSE_PAYLOAD_MODES + FINE_PAYLOAD_MODES + SEGMENT_PAYLOAD_MODES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=[
        "original", "SeaCache", "TeaCache", "SeaCachePayload", "TeaCachePayload",
        "TaylorSeer_fine", "HiCache_fine",
        "SeaCacheFinePayload", "SeaCacheSegmentPayload",
    ], default="original")
    p.add_argument("--task", default="t2v-1.3B", choices=["t2v-1.3B"])
    p.add_argument("--ckpt_dir", type=Path, default=os.environ.get("WAN21_CKPT_DIR"))
    p.add_argument(
        "--wan_repo",
        type=Path,
        default=_PROJECT_ROOT / "reference/taylorseer/code/TaylorSeer-Wan2.1",
        help="Directory containing the importable upstream `wan` package.",
    )
    p.add_argument("--prompt_file", type=Path, default=Path("resources/prompts/prompt.txt"))
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=65)
    p.add_argument("--sample_solver", choices=["unipc", "dpm++"], default="unipc")
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--dtype", choices=["bf16"], default="bf16")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--offload_model", action="store_true")
    p.add_argument("--t5_cpu", action="store_true")
    p.add_argument("--first_enhance", type=int, default=1)
    p.add_argument("--seacache_thresh", type=float, default=0.20)
    p.add_argument("--seacache_power_exp", type=float, default=3.0)
    p.add_argument("--seacache_norm_mode", choices=["mean", "peak"], default="mean")
    p.add_argument("--teacache_thresh", type=float, default=0.30)
    p.add_argument("--teacache_variant", default="1.3b")
    p.add_argument("--fresh_threshold", type=int, default=5)
    p.add_argument("--max_order", type=int, default=1)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--payload_mode", choices=PAYLOAD_MODES, default="reuse")
    p.add_argument("--payload_schedule_dir", type=Path, default=None)
    p.add_argument("--payload_sigma", type=float, default=0.5)
    p.add_argument("--payload_blend", type=float, default=1.0)
    p.add_argument("--segment_layout", choices=SEGMENT_LAYOUTS, default="seg8")
    p.add_argument("--require_locked_schedule", action="store_true")
    return p.parse_args()


def validate_protocol(args: argparse.Namespace) -> None:
    if args.ckpt_dir is None:
        raise SystemExit("--ckpt_dir is required or WAN21_CKPT_DIR must be set")
    if args.task != "t2v-1.3B":
        raise SystemExit("main Wan2.1 protocol requires --task t2v-1.3B")
    if (args.width, args.height) != (832, 480):
        raise SystemExit("main Wan2.1 protocol requires --width 832 --height 480")
    if args.num_frames != 65:
        raise SystemExit("main Wan2.1 protocol requires --num_frames 65")
    if args.num_steps != 50:
        raise SystemExit("main Wan2.1 protocol requires --num_steps 50")
    if args.sample_solver != "unipc":
        raise SystemExit("main Wan2.1 protocol requires --sample_solver unipc")
    if abs(float(args.sample_shift) - 5.0) > 1e-9:
        raise SystemExit("main Wan2.1 protocol requires --sample_shift 5.0")
    if abs(float(args.guidance_scale) - 5.0) > 1e-9:
        raise SystemExit("main Wan2.1 protocol requires --guidance_scale 5.0")
    payload_reuse_modes = {
        "SeaCachePayload": "reuse",
        "TeaCachePayload": "reuse",
        "SeaCacheFinePayload": "fine_reuse",
        "SeaCacheSegmentPayload": "segment_reuse",
    }
    if args.mode in payload_reuse_modes and args.payload_mode != payload_reuse_modes[args.mode]:
        if args.payload_schedule_dir is None:
            raise SystemExit("forecast payload modes require --payload_schedule_dir")
        if not args.require_locked_schedule:
            raise SystemExit("forecast payload modes require --require_locked_schedule")
    if args.payload_schedule_dir is not None and args.mode not in payload_reuse_modes:
        raise SystemExit("--payload_schedule_dir is only valid for payload modes")
    if args.mode in {"SeaCachePayload", "TeaCachePayload"} and args.payload_mode not in COARSE_PAYLOAD_MODES:
        raise SystemExit(f"{args.mode} requires coarse payload_mode in {COARSE_PAYLOAD_MODES}")
    if args.mode == "SeaCacheFinePayload" and args.payload_mode not in FINE_PAYLOAD_MODES:
        raise SystemExit(f"{args.mode} requires fine payload_mode in {FINE_PAYLOAD_MODES}")
    if args.mode == "SeaCacheSegmentPayload" and args.payload_mode not in SEGMENT_PAYLOAD_MODES:
        raise SystemExit(f"{args.mode} requires segment payload_mode in {SEGMENT_PAYLOAD_MODES}")
    if args.mode == "SeaCacheFinePayload" and abs(float(args.payload_blend) - 1.0) > 1e-9:
        raise SystemExit("SeaCacheFinePayload requires --payload_blend 1.0")
    if args.mode == "SeaCacheSegmentPayload" and abs(float(args.payload_blend) - 1.0) > 1e-9:
        raise SystemExit("SeaCacheSegmentPayload requires --payload_blend 1.0")
    if args.mode in {"TaylorSeer_fine", "HiCache_fine"}:
        if int(args.first_enhance) != 1:
            raise SystemExit(f"Wan2.1 {args.mode} baseline requires --first_enhance 1")
        if int(args.fresh_threshold) < 2:
            raise SystemExit(f"Wan2.1 {args.mode} baseline requires --fresh_threshold >= 2")
        if int(args.max_order) < 0:
            raise SystemExit(f"Wan2.1 {args.mode} baseline requires --max_order >= 0")
    if args.mode == "HiCache_fine" and not (0.0 < float(args.hicache_sigma) <= 1.0):
        raise SystemExit("Wan2.1 HiCache_fine baseline requires 0 < --hicache_sigma <= 1")


def import_wan(wan_repo: Path) -> Tuple[Any, Dict[str, Any], Dict[str, Tuple[int, int]], Any]:
    repo = Path(wan_repo).resolve()
    if not (repo / "wan").is_dir():
        raise SystemExit(f"wan repo does not contain a wan package: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_easydict_fallback()
    _install_xfuser_fallback()
    import wan  # type: ignore
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
    attention_backend = _require_flash_attention_backend()

    return wan, WAN_CONFIGS, SIZE_CONFIGS, attention_backend


def _install_easydict_fallback() -> None:
    try:
        import easydict  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class EasyDict(dict):
        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

        def __delattr__(self, name: str) -> None:
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    module = types.ModuleType("easydict")
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def _install_xfuser_fallback() -> None:
    try:
        import xfuser  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _SingleProcessGroup:
        def all_gather(self, x: Any, dim: int = 0) -> Any:
            return x

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_sequence_parallel_rank = lambda: 0
    distributed.get_sequence_parallel_world_size = lambda: 1
    distributed.get_sp_group = lambda: _SingleProcessGroup()
    distributed.initialize_model_parallel = lambda *args, **kwargs: None
    distributed.init_distributed_environment = lambda *args, **kwargs: None

    core = types.ModuleType("xfuser.core")
    core.distributed = distributed

    xfuser = types.ModuleType("xfuser")
    xfuser.core = core

    sys.modules["xfuser"] = xfuser
    sys.modules["xfuser.core"] = core
    sys.modules["xfuser.core.distributed"] = distributed


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_flash_attention_backend() -> Dict[str, Any]:
    from wan.modules import attention as attention_module  # type: ignore
    from wan.modules import model as model_module  # type: ignore

    flash_attn_2 = bool(getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False))
    flash_attn_3 = bool(getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False))
    if not (flash_attn_2 or flash_attn_3):
        raise RuntimeError(
            "Wan2.1 generation requires flash-attn. Refusing to use the "
            "scaled_dot_product_attention fallback; install flash-attn in the "
            "cache environment and rerun."
        )
    if getattr(model_module, "flash_attention", None) is not getattr(attention_module, "flash_attention", None):
        raise RuntimeError("Wan model flash_attention binding was modified before runner startup")

    return {
        "backend": "flash_attn_3" if flash_attn_3 else "flash_attn_2",
        "flash_attn_2_available": flash_attn_2,
        "flash_attn_3_available": flash_attn_3,
        "flash_attn_version": _package_version("flash-attn"),
        "sdpa_fallback_allowed": False,
    }


def _require_video_writer_backend() -> Dict[str, Any]:
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Wan2.1 video output requires imageio and imageio-ffmpeg. "
            "Refusing OpenCV/mp4v fallback."
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe or not Path(ffmpeg_exe).is_file():
        raise RuntimeError(f"imageio-ffmpeg did not return a valid ffmpeg binary: {ffmpeg_exe!r}")
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_exe)
    return {
        "backend": "imageio-ffmpeg",
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "ffmpeg_exe": ffmpeg_exe,
        "imageio_version": _package_version("imageio"),
        "imageio_ffmpeg_version": _package_version("imageio-ffmpeg"),
        "opencv_fallback_allowed": False,
    }


def _dtype_from_config(pipe: Any) -> Any:
    return pipe.param_dtype


def generate_t2v(
    pipe: Any,
    *,
    prompt: str,
    size: Tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    offload_model: bool,
) -> Tuple[Any, Dict[str, float]]:
    import torch
    import torch.cuda.amp as amp
    import torch.distributed as dist
    from tqdm import tqdm
    from wan.utils.fm_solvers import (  # type: ignore
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

    start_total = time.perf_counter()
    target_shape = (
        pipe.vae.model.z_dim,
        (frame_num - 1) // pipe.vae_stride[0] + 1,
        size[1] // pipe.vae_stride[1],
        size[0] // pipe.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (pipe.patch_size[1] * pipe.patch_size[2])
        * target_shape[1]
        / pipe.sp_size
    ) * pipe.sp_size

    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(int(seed))

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        context_null = pipe.text_encoder([pipe.sample_neg_prompt], pipe.device)
        if offload_model:
            pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([pipe.sample_neg_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=pipe.device,
            generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipe.model, "no_sync", noop_no_sync)
    denoise_start = time.perf_counter()
    with amp.autocast(dtype=_dtype_from_config(pipe)), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=pipe.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=pipe.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(sample_solver)

        pipe.model.scheduler = sample_scheduler
        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        pipe.model.to(pipe.device)
        for t in tqdm(timesteps, disable=os.environ.get("WAN21_TQDM", "0") != "1"):
            latent_model_input = latents
            timestep = torch.stack([t])
            noise_pred_cond = pipe.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipe.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        denoise_s = time.perf_counter() - denoise_start
        if offload_model:
            pipe.model.cpu()
            torch.cuda.empty_cache()

        decode_start = time.perf_counter()
        video = pipe.vae.decode(x0)[0]
        decode_s = time.perf_counter() - decode_start

    del noise, latents, sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return video, {
        "denoise_s": denoise_s,
        "decode_s": decode_s,
        "total_s": time.perf_counter() - start_total,
    }


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _video_complete(path: Path) -> bool:
    return Path(path).is_file() and Path(path).stat().st_size > 1024


def save_video_tensor(video: Any, save_file: Path, *, fps: int) -> None:
    import imageio.v2 as imageio
    import numpy as np
    import torch

    _require_video_writer_backend()
    tensor = video.detach()
    if tensor.ndim != 4:
        raise ValueError(f"expected decoded Wan video tensor [C,F,H,W], got shape {tuple(tensor.shape)}")
    tensor = tensor.to(torch.float32).clamp(-1.0, 1.0)
    frames = ((tensor + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()

    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_file.with_name(f"{save_file.stem}.tmp.{os.getpid()}.mp4")
    writer = imageio.get_writer(
        str(tmp),
        format="FFMPEG",
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=16,
        output_params=["-pix_fmt", "yuv420p"],
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()
    if not tmp.is_file() or tmp.stat().st_size <= 1024:
        raise RuntimeError(f"failed to write non-empty video: {tmp}")
    tmp.replace(save_file)


def main() -> None:
    configure_logging()
    args = parse_args()
    validate_protocol(args)
    args.prompt_file = Path(args.prompt_file)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wan, WAN_CONFIGS, SIZE_CONFIGS, attention_backend = import_wan(args.wan_repo)
    video_writer_backend = _require_video_writer_backend()
    prompts, start_idx, prompt_count = read_prompt_shard(
        args.prompt_file,
        limit=args.limit,
        shard_idx=args.shard_idx,
        shard_count=args.shard_count,
    )
    manifest = build_manifest(args, repo_root=_PROJECT_ROOT, prompt_count=prompt_count)
    manifest["attention_backend"] = attention_backend
    manifest["video_writer"] = video_writer_backend
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    atomic_write_json(
        args.output_dir / f"manifest_shard{args.shard_idx:03d}of{args.shard_count:03d}.json",
        manifest,
    )

    device_id = 0
    import torch

    torch.cuda.set_device(device_id)
    cfg = WAN_CONFIGS[args.task]
    LOG.info("Creating WanT2V: task=%s ckpt=%s", args.task, args.ckpt_dir)
    t_model_load = time.perf_counter()
    pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    model_load_s = time.perf_counter() - t_model_load

    if args.mode == "original":
        restore_original_forwards(pipe.model)
    elif args.mode == "TaylorSeer_fine":
        install_taylorseer_fine_forward(
            pipe.model,
            TaylorSeerFineConfig(
                num_steps=args.num_steps,
                fresh_threshold=args.fresh_threshold,
                first_enhance=args.first_enhance,
                max_order=args.max_order,
                mode="TaylorSeer_fine",
                basis="taylor",
                hicache_sigma=args.hicache_sigma,
            ),
        )
    elif args.mode == "HiCache_fine":
        install_hicache_fine_forward(
            pipe.model,
            TaylorSeerFineConfig(
                num_steps=args.num_steps,
                fresh_threshold=args.fresh_threshold,
                first_enhance=args.first_enhance,
                max_order=args.max_order,
                mode="HiCache_fine",
                basis="hicache",
                hicache_sigma=args.hicache_sigma,
            ),
        )
    elif args.mode == "SeaCacheFinePayload":
        install_fine_payload_forward(
            pipe.model,
            FinePayloadConfig(
                mode=args.mode,
                num_steps=args.num_steps,
                first_enhance=args.first_enhance,
                seacache_thresh=args.seacache_thresh,
                seacache_power_exp=args.seacache_power_exp,
                seacache_norm_mode=args.seacache_norm_mode,
                payload_mode=args.payload_mode,
                payload_sigma=args.payload_sigma,
                require_locked_schedule=args.require_locked_schedule,
            ),
        )
    elif args.mode == "SeaCacheSegmentPayload":
        install_segment_payload_forward(
            pipe.model,
            SegmentPayloadConfig(
                mode=args.mode,
                num_steps=args.num_steps,
                first_enhance=args.first_enhance,
                seacache_thresh=args.seacache_thresh,
                seacache_power_exp=args.seacache_power_exp,
                seacache_norm_mode=args.seacache_norm_mode,
                payload_mode=args.payload_mode,
                payload_sigma=args.payload_sigma,
                segment_layout=args.segment_layout,
                require_locked_schedule=args.require_locked_schedule,
            ),
        )
    else:
        cache_cfg = CacheForwardConfig(
            mode=args.mode,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            seacache_thresh=args.seacache_thresh,
            seacache_power_exp=args.seacache_power_exp,
            seacache_norm_mode=args.seacache_norm_mode,
            teacache_thresh=args.teacache_thresh,
            teacache_variant=args.teacache_variant,
            payload_mode=args.payload_mode,
            payload_sigma=args.payload_sigma,
            payload_blend=args.payload_blend,
            require_locked_schedule=args.require_locked_schedule,
        )
        install_cache_forward(pipe.model, cache_cfg)

    per_image = []
    wall_start = time.perf_counter()
    size = SIZE_CONFIGS[f"{args.width}*{args.height}"]
    for local_i, prompt in enumerate(prompts):
        global_idx = start_idx + local_i
        out_path = args.output_dir / video_filename(global_idx)
        dec_path = args.output_dir / decisions_filename(global_idx)
        if args.resume and _video_complete(out_path) and (args.mode == "original" or dec_path.exists()):
            LOG.info("resume skip idx=%d", global_idx)
            continue

        locked = None
        if args.payload_schedule_dir is not None:
            locked = load_locked_schedule(args.payload_schedule_dir, global_idx)

        if args.mode == "TaylorSeer_fine":
            reset_taylorseer_fine_state(
                pipe.model,
                prompt_idx=global_idx,
                seed=seed_for(args.seed, global_idx),
            )
        elif args.mode == "HiCache_fine":
            reset_hicache_fine_state(
                pipe.model,
                prompt_idx=global_idx,
                seed=seed_for(args.seed, global_idx),
            )
        elif args.mode == "SeaCacheFinePayload":
            reset_fine_payload_state(
                pipe.model,
                prompt_idx=global_idx,
                seed=seed_for(args.seed, global_idx),
                locked_schedule=locked,
            )
        elif args.mode == "SeaCacheSegmentPayload":
            reset_segment_payload_state(
                pipe.model,
                prompt_idx=global_idx,
                seed=seed_for(args.seed, global_idx),
                locked_schedule=locked,
            )
        elif args.mode != "original":
            reset_cache_state(
                pipe.model,
                prompt_idx=global_idx,
                seed=seed_for(args.seed, global_idx),
                locked_schedule=locked,
            )

        LOG.info("generate idx=%d seed=%d mode=%s", global_idx, seed_for(args.seed, global_idx), args.mode)
        video, timing = generate_t2v(
            pipe,
            prompt=prompt,
            size=size,
            frame_num=args.num_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.num_steps,
            guide_scale=args.guidance_scale,
            seed=seed_for(args.seed, global_idx),
            offload_model=args.offload_model,
        )
        save_video_tensor(video, out_path, fps=cfg.sample_fps)
        if args.mode != "original":
            if args.mode == "TaylorSeer_fine":
                dec = taylorseer_fine_decisions(pipe.model)
            elif args.mode == "HiCache_fine":
                dec = taylorseer_fine_decisions(pipe.model)
            elif args.mode == "SeaCacheFinePayload":
                dec = fine_payload_decisions(pipe.model)
            elif args.mode == "SeaCacheSegmentPayload":
                dec = segment_payload_decisions(pipe.model)
            else:
                dec = cache_decisions(pipe.model)
            dec.update({
                "prompt": prompt,
                "video_file": out_path.name,
                "task": args.task,
                "size": f"{args.width}*{args.height}",
                "num_frames": args.num_frames,
                "num_steps": args.num_steps,
                "sample_solver": args.sample_solver,
                "sample_shift": args.sample_shift,
                "guidance_scale": args.guidance_scale,
            })
            atomic_write_json(dec_path, dec)

        per_image.append({
            "idx": global_idx,
            "prompt": prompt,
            "seed": seed_for(args.seed, global_idx),
            "denoise_s": timing["denoise_s"],
            "decode_s": timing["decode_s"],
            "video_file": out_path.name,
            "decisions_file": dec_path.name if args.mode != "original" else None,
        })

    timing_config = {
        "cache_mode": args.mode,
        "mode": args.mode,
        "num_steps": args.num_steps,
        "num_frames": args.num_frames,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "seed_rule": "base_plus_prompt_idx",
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "seacache_thresh": args.seacache_thresh,
        "teacache_thresh": args.teacache_thresh,
        "fresh_threshold": args.fresh_threshold,
        "max_order": args.max_order,
        "hicache_sigma": args.hicache_sigma,
        "payload_mode": args.payload_mode,
        "payload_schedule_dir": None if args.payload_schedule_dir is None else str(args.payload_schedule_dir),
        "segment_layout": args.segment_layout,
    }
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    write_timing_json(
        args.output_dir / f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json",
        per_image=per_image,
        config=timing_config,
        model_load_s=model_load_s,
        wallclock_total_s=time.perf_counter() - wall_start + model_load_s,
        device=device_name,
    )


if __name__ == "__main__":
    main()
