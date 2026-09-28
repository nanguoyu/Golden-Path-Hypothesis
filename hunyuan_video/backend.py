from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from hunyuan_video.actions import FixedScheduleAction, FullAction, IntervalAction, evenly_spaced_cache_steps
from hunyuan_video.adapter import (
    CoarseBackboneAdapter,
    HiCacheFineAdapter,
    L2POutputAdapter,
    TaylorSeerFineAdapter,
)
from hunyuan_video.dicache import HunyuanDiCacheAdapter, HunyuanDiCacheConfig
from hunyuan_video.config import GenerationProtocol, RunSpec
from hunyuan_video.meancache_adapter import MeanCacheVelocityAdapter
from hunyuan_video.methods.reuse import ReuseMethod
from hunyuan_video.methods.hicache import HiCacheMethod
from hunyuan_video.methods.l2p import L2POutputMethod
from hunyuan_video.methods.meancache import MeanCacheMethod
from hunyuan_video.methods.seacache import SeaCacheMethod
from hunyuan_video.methods.sencache import SenCacheMethod
from hunyuan_video.methods.taylorseer import TaylorSeerMethod
from hunyuan_video.methods.teacache import TeaCacheMethod
from hunyuan_video.sencache import SenCacheAdapter


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = (ROOT / "reference/hunyuan_video/code").resolve()
OFFICIAL_COMMIT = "e748c73ac064728bf6bd15b1cdb8161e55a4f331"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_official_source() -> dict[str, Any]:
    if not (OFFICIAL_ROOT / "hyvideo/inference.py").is_file():
        raise FileNotFoundError(f"missing Tencent source: {OFFICIAL_ROOT}")
    return {"root": str(OFFICIAL_ROOT), "commit": None, "status_clean": None}


def _official_api(model_base: Path) -> dict[str, Any]:
    validate_official_source()
    os.environ["MODEL_BASE"] = str(model_base.resolve())
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    config = importlib.import_module("hyvideo.config")
    inference = importlib.import_module("hyvideo.inference")
    file_utils = importlib.import_module("hyvideo.utils.file_utils")
    for name, module in sys.modules.items():
        if name == "hyvideo" or name.startswith("hyvideo."):
            origin = getattr(module, "__file__", None)
            if origin and OFFICIAL_ROOT not in Path(origin).resolve().parents:
                raise RuntimeError(f"shadowed {name} imported from {origin}")
    return {
        "parse_args": config.parse_args,
        "sampler_class": inference.HunyuanVideoSampler,
        "save_videos_grid": file_utils.save_videos_grid,
    }


def _official_argv(model_base: Path, protocol: GenerationProtocol) -> list[str]:
    argv = [
        "--model", protocol.model,
        "--model-base", str(model_base),
        "--dit-weight", str(model_base / protocol.checkpoint),
        "--model-resolution", "720p",
        "--load-key", protocol.load_key,
        "--precision", protocol.precision,
        "--vae-precision", protocol.vae_precision,
        "--text-encoder-precision", protocol.text_encoder_precision,
        "--text-encoder-precision-2", protocol.text_encoder_precision_2,
        "--vae", "884-16c-hy",
        "--text-encoder", "llm",
        "--text-encoder-2", "clipL",
        "--prompt-template-video", protocol.prompt_template_video,
        "--hidden-state-skip-layer", str(protocol.hidden_state_skip_layer),
        "--video-size", str(protocol.height), str(protocol.width),
        "--video-length", str(protocol.frames),
        "--infer-steps", str(protocol.steps),
        "--flow-shift", str(protocol.flow_shift),
        "--flow-solver", protocol.flow_solver,
        "--cfg-scale", str(protocol.guidance_scale),
        "--embedded-cfg-scale", str(protocol.embedded_guidance_scale),
        "--ulysses-degree", "1",
        "--ring-degree", "1",
    ]
    if protocol.flow_reverse:
        argv.append("--flow-reverse")
    if protocol.apply_final_norm:
        argv.append("--apply-final-norm")
    if protocol.vae_tiling:
        argv.append("--vae-tiling")
    if protocol.cpu_offload:
        argv.append("--use-cpu-offload")
    return argv


def _parse_official_args(parse_args: Any, argv: list[str]) -> Any:
    original_argv = sys.argv
    try:
        sys.argv = ["hicache-hunyuan-stage-b", *argv]
        return parse_args()
    finally:
        sys.argv = original_argv


def _callable_identity(bound: Any) -> dict[str, Any]:
    function = getattr(bound, "__func__", bound)
    code = getattr(function, "__code__", None)
    source = inspect.getsourcefile(function)
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "source": str(Path(source).resolve()) if source else None,
        "code_sha256": hashlib.sha256(code.co_code).hexdigest() if code else None,
    }


def verify_untouched_transformer(transformer: Any) -> dict[str, Any]:
    modules = [transformer, *transformer.double_blocks, *transformer.single_blocks]
    if any("forward" in module.__dict__ for module in modules):
        raise RuntimeError("original mode found an instance-level forward replacement")
    hook_counts = {
        "pre": sum(len(module._forward_pre_hooks) for module in modules),
        "post": sum(len(module._forward_hooks) for module in modules),
    }
    if hook_counts != {"pre": 0, "post": 0}:
        raise RuntimeError(f"original mode found hooks: {hook_counts}")
    identity = _callable_identity(transformer.forward)
    if not identity["source"] or OFFICIAL_ROOT not in Path(identity["source"]).parents:
        raise RuntimeError(f"transformer forward is not Tencent source: {identity}")
    return {"forward": identity, "hooks": hook_counts, "instance_forward": False}


def load_official_sampler(
    model_base: Path, protocol: GenerationProtocol
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model_base = model_base.resolve()
    required = [
        model_base / protocol.checkpoint,
        model_base / "hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        model_base / "text_encoder/config.json",
        model_base / "text_encoder_2/config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete HunyuanVideo assets: {missing}")
    api = _official_api(model_base)
    argv = _official_argv(model_base, protocol)
    args = _parse_official_args(api["parse_args"], argv)
    started = time.perf_counter()
    sampler = api["sampler_class"].from_pretrained(model_base, args=args)
    load_seconds = time.perf_counter() - started
    identity = verify_untouched_transformer(sampler.pipeline.transformer)
    identity.update({"official_source": validate_official_source(), "official_argv": argv})
    return sampler, api, {"load_seconds": load_seconds, "identity": identity}


def inference_step_count(protocol: GenerationProtocol, run: RunSpec) -> int:
    configured = run.method_config.get("infer_steps")
    if configured is None:
        return protocol.steps
    if run.mode != "original":
        raise ValueError("infer_steps override is only valid for untouched original rows")
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise TypeError("infer_steps must be an integer")
    if not 1 <= configured <= protocol.steps:
        raise ValueError(f"infer_steps must be in [1, {protocol.steps}]")
    return configured


def prediction_kwargs(protocol: GenerationProtocol, run: RunSpec) -> dict[str, Any]:
    return {
        "prompt": run.prompt,
        "height": protocol.height,
        "width": protocol.width,
        "video_length": protocol.frames,
        "seed": run.seed,
        "infer_steps": inference_step_count(protocol, run),
        "guidance_scale": protocol.guidance_scale,
        "flow_shift": protocol.flow_shift,
        "embedded_guidance_scale": protocol.embedded_guidance_scale,
        "batch_size": 1,
        "num_videos_per_prompt": 1,
    }


def build_adapter(sampler: Any, protocol: GenerationProtocol, run: RunSpec) -> Any:
    transformer = sampler.pipeline.transformer
    config = run.method_config
    if run.mode == "reuse_interval":
        action = IntervalAction(
            protocol.steps,
            interval=int(config.get("interval", 5)),
            first_enhance=int(config.get("first_enhance", 1)),
            force_last=bool(config.get("force_last", True)),
        )
        return CoarseBackboneAdapter(transformer, ReuseMethod(action))
    if run.mode == "reuse_exact":
        raw_steps = config.get("cache_steps")
        if raw_steps is None:
            cache_steps = evenly_spaced_cache_steps(
                protocol.steps,
                int(config["cache_count"]),
            )
        else:
            if not isinstance(raw_steps, list) or any(
                isinstance(step, bool) or not isinstance(step, int)
                for step in raw_steps
            ):
                raise TypeError("reuse_exact cache_steps must be a list of integers")
            if raw_steps != sorted(set(raw_steps)):
                raise ValueError("reuse_exact cache_steps must be sorted and unique")
            if len(raw_steps) != int(config["cache_count"]):
                raise ValueError("reuse_exact cache_count differs from cache_steps")
            if raw_steps and (raw_steps[0] == 0 or raw_steps[-1] == protocol.steps - 1):
                raise ValueError("reuse_exact must keep first and terminal steps full")
            cache_steps = frozenset(raw_steps)
        return CoarseBackboneAdapter(
            transformer,
            ReuseMethod(FixedScheduleAction(protocol.steps, cache_steps)),
        )
    if run.mode == "golden_reuse_schedule":
        raw_steps = config.get("cache_steps")
        if not isinstance(raw_steps, list) or any(
            isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
        ):
            raise TypeError("golden_reuse_schedule cache_steps must be a list of integers")
        if raw_steps != sorted(set(raw_steps)):
            raise ValueError("golden_reuse_schedule cache_steps must be sorted and unique")
        if raw_steps and (raw_steps[0] == 0 or raw_steps[-1] == protocol.steps - 1):
            raise ValueError("golden_reuse_schedule must keep the first and last steps full")
        cache_count = config.get("cache_count")
        if isinstance(cache_count, bool) or not isinstance(cache_count, int):
            raise TypeError("golden_reuse_schedule cache_count must be an integer")
        if cache_count != len(raw_steps):
            raise ValueError("golden_reuse_schedule cache_count differs from cache_steps")
        return CoarseBackboneAdapter(
            transformer,
            ReuseMethod(FixedScheduleAction(protocol.steps, frozenset(raw_steps))),
        )
    if run.mode == "golden_taylor_schedule":
        raw_steps = config.get("cache_steps")
        if not isinstance(raw_steps, list) or any(
            isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
        ):
            raise TypeError("golden_taylor_schedule cache_steps must be a list of integers")
        if raw_steps != sorted(set(raw_steps)):
            raise ValueError("golden_taylor_schedule cache_steps must be sorted and unique")
        if raw_steps and (raw_steps[0] == 0 or raw_steps[-1] == protocol.steps - 1):
            raise ValueError("golden_taylor_schedule must keep the first and last steps full")
        cache_count = config.get("cache_count")
        if isinstance(cache_count, bool) or not isinstance(cache_count, int):
            raise TypeError("golden_taylor_schedule cache_count must be an integer")
        if cache_count != len(raw_steps):
            raise ValueError("golden_taylor_schedule cache_count differs from cache_steps")
        max_order = config.get("max_order", 1)
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order != 1:
            raise ValueError("golden_taylor_schedule freezes max_order=1")
        return TaylorSeerFineAdapter(
            transformer,
            TaylorSeerMethod(
                action=FixedScheduleAction(protocol.steps, frozenset(raw_steps)),
                max_order=1,
            ),
        )
    if run.mode == "reuse_all_full":
        return CoarseBackboneAdapter(transformer, ReuseMethod(FullAction(protocol.steps)))
    if run.mode == "teacache":
        method = TeaCacheMethod(
            num_steps=protocol.steps,
            threshold=float(config.get("threshold", 0.15)),
            first_enhance=int(config.get("first_enhance", 1)),
        )
        return CoarseBackboneAdapter(transformer, method)
    if run.mode == "seacache":
        method = SeaCacheMethod(
            num_steps=protocol.steps,
            threshold=float(config.get("threshold", 0.20)),
            first_enhance=int(config.get("first_enhance", 1)),
            scheduler_provider=lambda: sampler.pipeline.scheduler,
            power_exp=float(config.get("power_exp", 3.0)),
        )
        return CoarseBackboneAdapter(transformer, method)
    if run.mode == "taylorseer":
        action = IntervalAction(
            protocol.steps,
            interval=int(config.get("interval", 5)),
            first_enhance=int(config.get("first_enhance", 1)),
            force_last=bool(config.get("force_last", False)),
        )
        return TaylorSeerFineAdapter(
            transformer,
            TaylorSeerMethod(action=action, max_order=int(config.get("max_order", 1))),
            source_history_compat=bool(config.get("source_history_compat", False)),
        )
    if run.mode == "taylorseer_exact":
        raw_steps = config.get("cache_steps")
        if raw_steps is None:
            cache_steps = evenly_spaced_cache_steps(
                protocol.steps,
                int(config["cache_count"]),
            )
        else:
            if not isinstance(raw_steps, list) or any(
                isinstance(step, bool) or not isinstance(step, int)
                for step in raw_steps
            ):
                raise TypeError("taylorseer_exact cache_steps must be a list of integers")
            if raw_steps != sorted(set(raw_steps)):
                raise ValueError("taylorseer_exact cache_steps must be sorted and unique")
            if len(raw_steps) != int(config["cache_count"]):
                raise ValueError("taylorseer_exact cache_count differs from cache_steps")
            if raw_steps and (raw_steps[0] == 0 or raw_steps[-1] == protocol.steps - 1):
                raise ValueError("taylorseer_exact must keep first and terminal steps full")
            cache_steps = frozenset(raw_steps)
        return TaylorSeerFineAdapter(
            transformer,
            TaylorSeerMethod(
                action=FixedScheduleAction(protocol.steps, cache_steps),
                max_order=int(config.get("max_order", 1)),
                # the warmup order clamp's window; 3 is the image-side
                # convention (flux/runner.py TaylorSeer default) and matches
                # the official TaylorSeer-HunyuanVideo first_enhance guard
                first_enhance=3,
            ),
        )
    if run.mode == "hicache_exact":
        raw_steps = config.get("cache_steps")
        first_enhance = int(config.get("first_enhance", 3))
        if raw_steps is None:
            action = IntervalAction(
                protocol.steps,
                interval=int(config["interval"]),
                first_enhance=first_enhance,
                force_last=True,
            )
        else:
            if not isinstance(raw_steps, list) or any(
                isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
            ):
                raise TypeError("hicache_exact cache_steps must be a list of integers")
            if raw_steps != sorted(set(raw_steps)):
                raise ValueError("hicache_exact cache_steps must be sorted and unique")
            if len(raw_steps) != int(config["cache_count"]):
                raise ValueError("hicache_exact cache_count differs from cache_steps")
            forbidden = set(range(first_enhance)) | {protocol.steps - 1}
            if forbidden.intersection(raw_steps):
                raise ValueError("hicache_exact caches a warmup or terminal step")
            action = FixedScheduleAction(protocol.steps, frozenset(raw_steps))
        return HiCacheFineAdapter(
            transformer,
            HiCacheMethod(
                action=action,
                max_order=int(config.get("max_order", 2)),
                sigma=float(config.get("sigma", 0.5)),
                # the same value that decides the forbidden warmup steps also
                # bounds the predictor's history depth during them
                first_enhance=first_enhance,
            ),
        )
    if run.mode == "l2p_output_exact":
        raw_steps = config.get("cache_steps")
        if not isinstance(raw_steps, list) or any(
            isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
        ):
            raise TypeError("l2p_output_exact cache_steps must be a list of integers")
        if raw_steps != sorted(set(raw_steps)):
            raise ValueError("l2p_output_exact cache_steps must be sorted and unique")
        if len(raw_steps) != int(config["cache_count"]):
            raise ValueError("l2p_output_exact cache_count differs from cache_steps")
        if 0 in raw_steps:
            raise ValueError("l2p_output_exact cannot cache step 0")
        return L2POutputAdapter(
            transformer,
            L2POutputMethod(
                action=FixedScheduleAction(protocol.steps, frozenset(raw_steps)),
                weights_path=str(config["weights_path"]),
                num_steps=protocol.steps,
                min_abs_weight=float(config.get("min_abs_weight", 0.0)),
            ),
        )
    if run.mode == "meancache_exact":
        raw_steps = config.get("cache_steps")
        if not isinstance(raw_steps, list) or any(
            isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
        ):
            raise TypeError("meancache_exact cache_steps must be a list of integers")
        if raw_steps != sorted(set(raw_steps)):
            raise ValueError("meancache_exact cache_steps must be sorted and unique")
        if len(raw_steps) != int(config["cache_count"]):
            raise ValueError("meancache_exact cache_count differs from cache_steps")
        # The path search forces first_full_steps=5 / last_full_steps=1 (plan
        # section 2.4 item 4), and the method raises rather than promoting a
        # cache step reached before its first full velocity, so a table the
        # search could not have emitted fails here instead of on the GPU.
        # `first_full_steps` is that 5 unless the caller lowered it: the video
        # SPX experiment transplants foreign schedules onto this payload and
        # runs it at 2 (`--spx_relax_warmup`), which is the smallest value at
        # which the JVP reference still lands on a step that has a velocity.
        first_full = int(config.get("first_full_steps", 5))
        if first_full < 2:
            raise ValueError("meancache_exact needs at least two full steps at the head")
        forbidden = set(range(first_full)) | {protocol.steps - 1}
        if forbidden.intersection(raw_steps):
            raise ValueError("meancache_exact caches a step the path search keeps full")
        raw_spans = config.get("jvp_spans") or {}
        if not isinstance(raw_spans, dict):
            raise TypeError("meancache_exact jvp_spans must be a mapping")
        return MeanCacheVelocityAdapter(
            transformer,
            MeanCacheMethod(
                action=FixedScheduleAction(protocol.steps, frozenset(raw_steps)),
                num_steps=protocol.steps,
                # `predict()` rebuilds the scheduler per call
                # (`hyvideo/inference.py:611-616`), so the sigmas have to be read
                # through the pipeline at step time, not bound here.
                scheduler_provider=lambda: sampler.pipeline.scheduler,
                jvp_span=int(config.get("jvp_span", 4)),
                jvp_spans={int(step): int(span) for step, span in raw_spans.items()},
            ),
        )
    if run.mode == "sencache":
        method = SenCacheMethod(
            num_steps=protocol.steps,
            sensitivity_path=str(config["sensitivity_path"]),
            threshold_start=float(config["threshold_start"]),
            threshold_main=float(config["threshold"]),
            first_enhance=int(config.get("first_enhance", 3)),
            max_skip=int(config.get("max_skip", 10)),
            switch_ratio=float(config.get("switch_ratio", 0.2)),
            ret_steps=int(config.get("ret_steps", 0)),
            cutoff_steps=int(config.get("cutoff_steps", -1)),
        )
        return SenCacheAdapter(transformer, method)
    if run.mode == "dicache":
        # With `cache_steps` the gate is replaced by the table and the payload
        # -- shallow probe, gamma clamp, two-anchor extrapolation -- is
        # unchanged; that is the video SPX `di_two_anchor` column, which is the
        # only way DiCache's payload can be scored on a foreign schedule.
        raw_steps = config.get("cache_steps")
        fixed_steps = None
        if raw_steps is not None:
            if not isinstance(raw_steps, list) or any(
                isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
            ):
                raise TypeError("dicache cache_steps must be a list of integers")
            if raw_steps != sorted(set(raw_steps)):
                raise ValueError("dicache cache_steps must be sorted and unique")
            if "cache_count" in config and len(raw_steps) != int(config["cache_count"]):
                raise ValueError("dicache cache_count differs from cache_steps")
            forbidden = {0, 1, protocol.steps - 1}
            if forbidden.intersection(raw_steps):
                raise ValueError(
                    "a fixed-schedule dicache run keeps steps 0, 1 and the terminal step "
                    "full: the two-anchor payload needs two anchors before its first "
                    "cached step")
            fixed_steps = tuple(int(step) for step in raw_steps)
        return HunyuanDiCacheAdapter(
            transformer,
            HunyuanDiCacheConfig(
                num_steps=protocol.steps,
                threshold=float(config.get("threshold", 0.0)),
                ret_ratio=float(config.get("ret_ratio", 0.2)),
                probe_depth=int(config.get("probe_depth", 1)),
                cache_steps=fixed_steps,
            ),
        )
    raise KeyError(f"unsupported HunyuanVideo mode: {run.mode}")


@contextlib.contextmanager
def maybe_adapter(sampler: Any, protocol: GenerationProtocol, run: RunSpec) -> Iterator[Any | None]:
    if run.mode == "original":
        verify_untouched_transformer(sampler.pipeline.transformer)
        yield None
        verify_untouched_transformer(sampler.pipeline.transformer)
        return
    adapter = build_adapter(sampler, protocol, run)
    adapter.reset()
    with adapter:
        yield adapter
    verify_untouched_transformer(sampler.pipeline.transformer)


def generate(sampler: Any, protocol: GenerationProtocol, run: RunSpec) -> tuple[dict[str, Any], Any | None, float, int]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with maybe_adapter(sampler, protocol, run) as adapter:
        output = sampler.predict(**prediction_kwargs(protocol, run))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0
    return output, adapter, time.perf_counter() - started, peak


def save_video(api: dict[str, Any], sample: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    api["save_videos_grid"](sample.unsqueeze(0), str(path), fps=fps)
