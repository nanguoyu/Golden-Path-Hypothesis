#!/usr/bin/env python3
"""Probe RFC feature-level assumptions on full FLUX trajectories.

This runner does not cache.  It installs the existing SeaCacheFinePayload
114-slot block hooks only as an observer boundary, runs full FLUX denoising,
and writes per-slot feature-change statistics:

* direction stability of consecutive input/output deltas,
* input-output delta alignment and magnitude ratio,
* Taylor relative input/output prediction errors for RCS auditing.

The output is intended for mechanism analysis, not timing/speedup claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.gates import rel_l1  # noqa: E402
from lib.io_utils import read_prompts, split_shard  # noqa: E402
from lib.taylor import taylor_predict  # noqa: E402

EPS = 1e-12

ROW_FIELDS = [
    "prompt_idx",
    "step",
    "timestep",
    "block_idx",
    "block_type",
    "slot",
    "slot_family",
    "input_norm",
    "output_norm",
    "delta_input_norm",
    "delta_output_norm",
    "s_ratio",
    "cos_delta_input_output",
    "cos_delta_input_prev",
    "cos_delta_output_prev",
    "input_mag_ratio_prev",
    "output_mag_ratio_prev",
    "input_mag_log_ratio_abs",
    "output_mag_log_ratio_abs",
    "input_hist_order_pre",
    "output_hist_order_pre",
    "rel_input_err_o1",
    "rel_output_taylor_err_o1",
    "rel_output_rfe_err_o1",
    "taylor_err_ratio_o1",
    "rfe_err_ratio_o1",
    "rel_input_err_o2",
    "rel_output_taylor_err_o2",
    "rel_output_rfe_err_o2",
    "taylor_err_ratio_o2",
    "rfe_err_ratio_o2",
]


def _to_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.10g}"


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).norm().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if tuple(a.shape) != tuple(b.shape):
        return None
    af = a.detach().to(torch.float32)
    bf = b.detach().to(torch.float32)
    an = float(af.norm().item())
    bn = float(bf.norm().item())
    if an <= 0.0 or bn <= 0.0:
        return None
    return float(torch.sum(af * bf).item()) / (an * bn + EPS)


def _mag_ratio(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None or curr <= 0.0 or prev <= 0.0:
        return None
    return float(curr / (prev + EPS))


def _log_ratio_abs(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    ratio = _mag_ratio(curr, prev)
    if ratio is None or ratio <= 0.0:
        return None
    return float(abs(math.log(ratio)))


def _history_order(history: Optional[Dict[int, torch.Tensor]]) -> int:
    if not isinstance(history, dict) or not history:
        return -1
    return max(int(k) for k in history)


class RfcAssumptionObserver:
    def __init__(self, rows_path: Path, max_order: int = 2):
        self.rows_path = Path(rows_path)
        self.max_order = int(max_order)
        self.rows_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.rows_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self.prompt_idx: Optional[int] = None
        self.prompt: Optional[str] = None
        self.step: Optional[int] = None
        self.timestep: Optional[float] = None
        self.row_count = 0
        self.prompt_row_counts: Dict[int, int] = {}

    def close(self) -> None:
        self._fh.close()

    def reset_prompt(self, prompt_idx: int, prompt: str) -> None:
        self.prompt_idx = int(prompt_idx)
        self.prompt = str(prompt)
        self.step = None
        self.timestep = None
        self.prompt_row_counts.setdefault(int(prompt_idx), 0)

    def begin_step(self, step: int, timestep: Any) -> None:
        self.step = int(step)
        self.timestep = self._scalar_timestep(timestep)

    @staticmethod
    def _scalar_timestep(timestep: Any) -> Optional[float]:
        if timestep is None:
            return None
        try:
            if isinstance(timestep, torch.Tensor):
                return float(timestep.detach().flatten()[0].to(torch.float32).item())
            return float(timestep)
        except Exception:
            return None

    def observe_slot(
        self,
        *,
        state: Any,
        key: Tuple[int, str, str],
        input_feature: torch.Tensor,
        output_feature: torch.Tensor,
        prev_input_history: Optional[Dict[int, torch.Tensor]],
        prev_output_history: Optional[Dict[int, torch.Tensor]],
    ) -> None:
        prompt_idx = state.prompt_idx if getattr(state, "prompt_idx", None) is not None else self.prompt_idx
        step = int(getattr(state, "cnt", self.step if self.step is not None else -1))
        block_idx, block_type, slot = key
        slot_family = f"{block_type}_{slot}"

        input_norm = _norm(input_feature)
        output_norm = _norm(output_feature)
        input_hist_order = _history_order(prev_input_history)
        output_hist_order = _history_order(prev_output_history)

        delta_input_norm = None
        delta_output_norm = None
        s_ratio = None
        cos_delta_input_output = None
        cos_delta_input_prev = None
        cos_delta_output_prev = None
        input_mag_ratio_prev = None
        output_mag_ratio_prev = None
        input_mag_log_ratio_abs = None
        output_mag_log_ratio_abs = None

        if (
            isinstance(prev_input_history, dict)
            and isinstance(prev_output_history, dict)
            and 0 in prev_input_history
            and 0 in prev_output_history
            and tuple(prev_input_history[0].shape) == tuple(input_feature.shape)
            and tuple(prev_output_history[0].shape) == tuple(output_feature.shape)
        ):
            d_input = input_feature.detach().to(torch.float32) - prev_input_history[0].detach().to(torch.float32)
            d_output = output_feature.detach().to(torch.float32) - prev_output_history[0].detach().to(torch.float32)
            delta_input_norm = float(d_input.norm().item())
            delta_output_norm = float(d_output.norm().item())
            if delta_input_norm > 0.0 and delta_output_norm > 0.0:
                s_ratio = float(delta_output_norm / (delta_input_norm + EPS))
                cos_delta_input_output = _cos(d_input, d_output)
            if 1 in prev_input_history and tuple(prev_input_history[1].shape) == tuple(d_input.shape):
                prev_d_input_norm = _norm(prev_input_history[1])
                cos_delta_input_prev = _cos(d_input, prev_input_history[1])
                input_mag_ratio_prev = _mag_ratio(delta_input_norm, prev_d_input_norm)
                input_mag_log_ratio_abs = _log_ratio_abs(delta_input_norm, prev_d_input_norm)
            if 1 in prev_output_history and tuple(prev_output_history[1].shape) == tuple(d_output.shape):
                prev_d_output_norm = _norm(prev_output_history[1])
                cos_delta_output_prev = _cos(d_output, prev_output_history[1])
                output_mag_ratio_prev = _mag_ratio(delta_output_norm, prev_d_output_norm)
                output_mag_log_ratio_abs = _log_ratio_abs(delta_output_norm, prev_d_output_norm)

        err_fields: Dict[str, Optional[float]] = {
            "rel_input_err_o1": None,
            "rel_output_taylor_err_o1": None,
            "rel_output_rfe_err_o1": None,
            "taylor_err_ratio_o1": None,
            "rfe_err_ratio_o1": None,
            "rel_input_err_o2": None,
            "rel_output_taylor_err_o2": None,
            "rel_output_rfe_err_o2": None,
            "taylor_err_ratio_o2": None,
            "rfe_err_ratio_o2": None,
        }
        for order in (1, 2):
            if (
                input_hist_order >= order
                and output_hist_order >= order
                and isinstance(prev_input_history, dict)
                and isinstance(prev_output_history, dict)
            ):
                pred_input = taylor_predict(prev_input_history, 1, order)
                pred_output = taylor_predict(prev_output_history, 1, order)
                if tuple(pred_input.shape) == tuple(input_feature.shape):
                    rel_input_err = rel_l1(pred_input, input_feature)
                else:
                    rel_input_err = None
                if tuple(pred_output.shape) == tuple(output_feature.shape):
                    rel_output_taylor_err = rel_l1(pred_output, output_feature)
                else:
                    rel_output_taylor_err = None
                rel_output_rfe_err = None
                if (
                    rel_output_taylor_err is not None
                    and isinstance(prev_input_history, dict)
                    and isinstance(prev_output_history, dict)
                    and 1 in prev_input_history
                    and 1 in prev_output_history
                    and tuple(prev_input_history[0].shape) == tuple(input_feature.shape)
                    and tuple(prev_output_history[0].shape) == tuple(output_feature.shape)
                ):
                    reuse = prev_output_history[0]
                    direction = pred_output.detach().to(torch.float32) - reuse.detach().to(torch.float32)
                    direction_norm = float(direction.norm().item())
                    input_delta = input_feature.detach().to(torch.float32) - prev_input_history[0].detach().to(torch.float32)
                    input_delta_norm = float(input_delta.norm().item())
                    hist_input_norm = _norm(prev_input_history[1])
                    hist_output_norm = _norm(prev_output_history[1])
                    if direction_norm > 0.0 and hist_input_norm > 0.0:
                        magnitude = hist_output_norm / (hist_input_norm + EPS) * input_delta_norm
                        pred_rfe = reuse.detach().to(torch.float32) + direction / (direction_norm + EPS) * magnitude
                        rel_output_rfe_err = rel_l1(pred_rfe, output_feature.detach().to(torch.float32))
                taylor_err_ratio = (
                    None
                    if rel_input_err is None or rel_input_err <= 0.0 or rel_output_taylor_err is None
                    else float(rel_output_taylor_err / (rel_input_err + EPS))
                )
                rfe_err_ratio = (
                    None
                    if rel_input_err is None or rel_input_err <= 0.0 or rel_output_rfe_err is None
                    else float(rel_output_rfe_err / (rel_input_err + EPS))
                )
                err_fields[f"rel_input_err_o{order}"] = rel_input_err
                err_fields[f"rel_output_taylor_err_o{order}"] = rel_output_taylor_err
                err_fields[f"rel_output_rfe_err_o{order}"] = rel_output_rfe_err
                err_fields[f"taylor_err_ratio_o{order}"] = taylor_err_ratio
                err_fields[f"rfe_err_ratio_o{order}"] = rfe_err_ratio

        row = {
            "prompt_idx": int(prompt_idx) if prompt_idx is not None else "",
            "step": step,
            "timestep": _to_float(self.timestep),
            "block_idx": int(block_idx),
            "block_type": str(block_type),
            "slot": str(slot),
            "slot_family": slot_family,
            "input_norm": _to_float(input_norm),
            "output_norm": _to_float(output_norm),
            "delta_input_norm": _to_float(delta_input_norm),
            "delta_output_norm": _to_float(delta_output_norm),
            "s_ratio": _to_float(s_ratio),
            "cos_delta_input_output": _to_float(cos_delta_input_output),
            "cos_delta_input_prev": _to_float(cos_delta_input_prev),
            "cos_delta_output_prev": _to_float(cos_delta_output_prev),
            "input_mag_ratio_prev": _to_float(input_mag_ratio_prev),
            "output_mag_ratio_prev": _to_float(output_mag_ratio_prev),
            "input_mag_log_ratio_abs": _to_float(input_mag_log_ratio_abs),
            "output_mag_log_ratio_abs": _to_float(output_mag_log_ratio_abs),
            "input_hist_order_pre": input_hist_order,
            "output_hist_order_pre": output_hist_order,
            **{key: _to_float(value) for key, value in err_fields.items()},
        }
        self._writer.writerow(row)
        self.row_count += 1
        if prompt_idx is not None:
            self.prompt_row_counts[int(prompt_idx)] = self.prompt_row_counts.get(int(prompt_idx), 0) + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe RFC feature assumptions on full FLUX trajectories.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--max_order", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _patch_transformer_forward(pipe, *, max_order: int, num_steps: int) -> Callable[[], None]:
    from diffusers.models import FluxTransformer2DModel

    original_forward = FluxTransformer2DModel.forward

    def probe_forward(self, *args, **kwargs):
        state = getattr(self, "_seacache_fine_payload_state", None)
        observer = getattr(state, "observer", None) if state is not None else None
        if state is None or observer is None:
            return original_forward(self, *args, **kwargs)
        cnt = int(state.cnt)
        state.reset_step_accounting()
        state.should_skip = False
        state.step_offset = 0
        state.step_gap = 1
        state.effective_max_order = int(max_order)
        observer.begin_step(cnt, kwargs.get("timestep"))
        out = original_forward(self, *args, **kwargs)
        state.previous_full_step = state.last_full_step
        state.last_full_step = cnt
        state.full_steps_seen += 1
        state.cnt = (cnt + 1) % int(num_steps)
        return out

    FluxTransformer2DModel.forward = probe_forward

    def teardown() -> None:
        FluxTransformer2DModel.forward = original_forward

    return teardown


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    rows_path = args.output_dir / f"rfc_assumption_rows_shard{args.shard_idx}of{args.shard_count}.csv"
    done_path = args.output_dir / f"rfc_assumption_shard{args.shard_idx}of{args.shard_count}.done.json"
    if args.resume and done_path.is_file():
        print(f"[shard {args.shard_idx}] done marker exists, skipping: {done_path}", flush=True)
        return 0
    if not shard_prompts:
        _write_summary(done_path, {"empty": True, "shard_idx": args.shard_idx, "shard_count": args.shard_count})
        return 0

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} dtype={args.dtype} "
        f"for RFC assumption probe shard {args.shard_idx}/{args.shard_count}",
        flush=True,
    )
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    from flux import seacache_fine_payload

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in {time.perf_counter() - process_start:.1f}s; "
        f"global idx {start}..{end - 1}",
        flush=True,
    )

    block_teardown = seacache_fine_payload.install_block_hooks(
        pipe,
        threshold=0.0,
        num_steps=int(args.num_steps),
        first_enhance=0,
        payload_mode="fine_rfc_rfe_taylor_o2" if int(args.max_order) >= 2 else "fine_rfc_rfe_taylor_o1",
        payload_sigma=0.5,
        payload_gate_mode="seacache",
    )
    transformer_teardown = _patch_transformer_forward(
        pipe,
        max_order=int(args.max_order),
        num_steps=int(args.num_steps),
    )
    state = pipe.transformer._seacache_fine_payload_state
    observer = RfcAssumptionObserver(rows_path, max_order=int(args.max_order))
    state.observer = observer

    per_prompt = []
    try:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = start + local_idx
            per_image_seed = int(args.seed) + int(global_idx)
            generator = torch.Generator(device=device).manual_seed(per_image_seed)
            state.reset_trajectory(prompt_idx=global_idx)
            state.observer = observer
            observer.reset_prompt(global_idx, prompt)
            t0 = time.perf_counter()
            _ = pipe(
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
            elapsed = time.perf_counter() - t0
            per_prompt.append({
                "prompt_idx": int(global_idx),
                "seed": int(per_image_seed),
                "elapsed_s": float(elapsed),
                "rows": int(observer.prompt_row_counts.get(int(global_idx), 0)),
            })
            print(
                f"[shard {args.shard_idx}] prompt {global_idx} rows={per_prompt[-1]['rows']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        observer.close()
        try:
            state.observer = None
        except Exception:
            pass
        transformer_teardown()
        block_teardown()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_file": str(args.prompt_file),
        "output_dir": str(args.output_dir),
        "rows_path": str(rows_path),
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "prompt_start": int(start),
        "prompt_end": int(end),
        "n_prompts": int(len(shard_prompts)),
        "num_steps": int(args.num_steps),
        "expected_slots": int(state.expected_slots),
        "max_order": int(args.max_order),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "model_id": str(args.model_id),
        "row_count": int(observer.row_count),
        "per_prompt": per_prompt,
        "elapsed_s": float(time.perf_counter() - process_start),
    }
    _write_summary(done_path, summary)
    print(f"[DONE] wrote {observer.row_count} rows -> {rows_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
