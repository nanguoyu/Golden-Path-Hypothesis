#!/usr/bin/env python3
"""Exhaustively score the frozen FLUX K=41 residual-reuse schedule universe.

The model stays resident on one GPU.  Each schedule is evaluated on the same
four prompt--seed pairs against full-compute references decoded by the same
pipeline.  Only scalar terminal errors are stored; images and step decisions
are intentionally not written during the exhaustive pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.exhaustive_schedule import (  # noqa: E402
    K41_SPACE,
    iter_lex_range,
    mask_hex,
    schedule_for_rank,
    split_rank_interval,
)
from lib.io_utils import read_prompts  # noqa: E402


PART_SCHEMA = "flux_exhaustive_k41_part.v1"
WORKER_SCHEMA = "flux_exhaustive_k41_worker.v1"
DEFAULT_PROMPT_INDICES = (5, 8, 9, 15)
CONDITIONING_SCHEMA = "flux_exhaustive_k41_conditioning.v1"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _parse_indices(value: str, total: int) -> tuple[int, ...]:
    try:
        indices = tuple(
            int(piece.strip()) for piece in value.split(",") if piece.strip()
        )
    except ValueError as exc:
        raise SystemExit(f"invalid --prompt_indices: {value!r}") from exc
    if not indices or len(set(indices)) != len(indices):
        raise SystemExit(
            "--prompt_indices must contain distinct comma-separated integers"
        )
    if min(indices) < 0 or max(indices) >= total:
        raise SystemExit(f"prompt indices {indices} outside [0, {total})")
    return indices


def _image_array(image: Any) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"decoded image has unexpected shape {array.shape}")
    return array


def mse_and_psnr(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Standard uint8 RGB MSE and PSNR (data range 255).

    Exact matches receive the finite sentinel 120 dB so the JSON remains
    standards-compliant.  K=41 should not produce an exact full-compute image;
    a separate ``zero_mse`` field records the event if it ever occurs.
    """

    if reference.shape != candidate.shape:
        raise ValueError(f"image shapes differ: {reference.shape} vs {candidate.shape}")
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(diff * diff, dtype=np.float64))
    if mse == 0.0:
        return 0.0, 120.0
    return mse, float(10.0 * math.log10((255.0 * 255.0) / mse))


def _prompt_text_sha256(selected_prompts: tuple[str, ...]) -> str:
    prompt_bytes = json.dumps(
        list(selected_prompts), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(prompt_bytes).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_payload(
    args: argparse.Namespace,
    prompt_indices: tuple[int, ...],
    selected_prompts: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema": "flux_exhaustive_k41_experiment.v1",
        "space_identity": K41_SPACE.identity,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_name": args.model_name,
        "num_steps": int(args.num_steps),
        "width": int(args.width),
        "height": int(args.height),
        "guidance": float(args.guidance),
        "dtype": args.dtype,
        "prompt_file": str(args.prompt_file),
        "prompt_indices": list(prompt_indices),
        "prompt_text_sha256": _prompt_text_sha256(selected_prompts),
        "base_seed": int(args.seed),
        "seed_rule": "base_plus_global_prompt_index",
        "metric": "uint8_rgb_psnr_data_range_255",
        "precompute_prompt_embeddings": bool(args.precompute_prompt_embeddings),
        "conditioning_artifact_sha256": args.conditioning_artifact_sha256,
        "conditioning_mode": (
            "frozen_artifact"
            if args.conditioning_file is not None
            else "runtime_text_encoder"
        ),
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_existing_part(
    path: Path,
    *,
    rank_start: int,
    rank_end: int,
    experiment_fingerprint: str,
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resume part is unreadable: {path}") from exc
    if payload.get("schema") != PART_SCHEMA:
        raise RuntimeError(f"resume part has wrong schema: {path}")
    if payload.get("experiment_fingerprint") != experiment_fingerprint:
        raise RuntimeError(f"resume part belongs to another experiment: {path}")
    if (payload.get("rank_start"), payload.get("rank_end")) != (rank_start, rank_end):
        raise RuntimeError(f"resume part has wrong rank interval: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or [row.get("rank") for row in rows] != list(
        range(rank_start, rank_end)
    ):
        raise RuntimeError(f"resume part is incomplete or out of order: {path}")


def _encode_prompt(pipe: Any, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    """Encode immutable text conditioning once per worker and prompt."""

    import torch

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.device,
            num_images_per_prompt=1,
            max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        )
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def _load_conditioning_artifact(
    path: Path,
    *,
    pipe: Any,
    args: argparse.Namespace,
    prompt_indices: tuple[int, ...],
    selected_prompts: tuple[str, ...],
    resolved_model_commit: str | None,
) -> dict[int, dict[str, Any]]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("schema") != CONDITIONING_SCHEMA:
        raise ValueError(f"wrong conditioning artifact schema: {path}")
    expected_prompt_sha = _prompt_text_sha256(selected_prompts)
    checks = {
        "model_id": (artifact.get("model_id"), args.model_id),
        "model_revision": (artifact.get("model_revision"), args.revision),
        "prompt_indices": (artifact.get("prompt_indices"), list(prompt_indices)),
        "prompt_text_sha256": (
            artifact.get("prompt_text_sha256"),
            expected_prompt_sha,
        ),
        "max_sequence_length": (
            artifact.get("max_sequence_length"),
            256 if args.model_name == "flux-schnell" else 512,
        ),
    }
    mismatches = {
        key: {"artifact": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    artifact_commit = artifact.get("model_commit")
    if resolved_model_commit is not None and artifact_commit != resolved_model_commit:
        mismatches["model_commit"] = {
            "artifact": artifact_commit,
            "expected": resolved_model_commit,
        }
    if mismatches:
        raise ValueError(f"conditioning artifact identity mismatch: {mismatches}")
    records = artifact.get("conditioning")
    if not isinstance(records, dict):
        raise ValueError("conditioning artifact has no conditioning records")
    loaded: dict[int, dict[str, Any]] = {}
    for prompt_idx in prompt_indices:
        record = records.get(str(prompt_idx))
        if not isinstance(record, dict):
            raise ValueError(f"conditioning artifact omits prompt {prompt_idx}")
        prompt_embeds = record.get("prompt_embeds")
        pooled_prompt_embeds = record.get("pooled_prompt_embeds")
        if not isinstance(prompt_embeds, torch.Tensor) or not isinstance(
            pooled_prompt_embeds, torch.Tensor
        ):
            raise ValueError(
                f"conditioning artifact prompt {prompt_idx} has bad tensors"
            )
        if (
            prompt_embeds.ndim != 3
            or prompt_embeds.shape[0] != 1
            or prompt_embeds.shape[1] != checks["max_sequence_length"][1]
            or pooled_prompt_embeds.ndim != 2
            or pooled_prompt_embeds.shape[0] != 1
            or prompt_embeds.dtype != torch.bfloat16
            or pooled_prompt_embeds.dtype != torch.bfloat16
        ):
            raise ValueError(
                f"conditioning artifact prompt {prompt_idx} has unexpected "
                f"shapes/dtypes: prompt={prompt_embeds.shape}/{prompt_embeds.dtype}, "
                f"pooled={pooled_prompt_embeds.shape}/{pooled_prompt_embeds.dtype}"
            )
        loaded[prompt_idx] = {
            "prompt_embeds": prompt_embeds.to(pipe.device),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(pipe.device),
        }
    return loaded


def _run_conditioned(
    pipe: Any, conditioning: dict[str, Any], seed: int, args: argparse.Namespace
):
    """Pipeline call equivalent to the string-prompt path, without re-encoding text."""

    import torch

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    result = pipe(
        prompt=None,
        prompt_2=None,
        prompt_embeds=conditioning["prompt_embeds"],
        pooled_prompt_embeds=conditioning["pooled_prompt_embeds"],
        num_inference_steps=int(args.num_steps),
        guidance_scale=(
            0.0 if args.model_name == "flux-schnell" else float(args.guidance)
        ),
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result.images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=Path("resources/prompts/partiprompts_full_eval1632_seed42.txt"),
    )
    parser.add_argument(
        "--prompt_indices",
        default=",".join(str(v) for v in DEFAULT_PROMPT_INDICES),
        help="comma-separated global indices; frozen default is 5,8,9,15",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--revision",
        default="3de623fc",
        help="FLUX snapshot pin; pass an empty string only for an explicitly unpinned smoke",
    )
    parser.add_argument(
        "--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev"
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--rank_start", type=int, default=0)
    parser.add_argument("--rank_end", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--conditioning_file",
        type=Path,
        default=None,
        help="frozen canonical prompt embeddings; required for formal multi-site runs",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow_protocol_override",
        action="store_true",
        help="smoke only; permit settings other than the frozen discovery protocol",
    )
    parser.set_defaults(precompute_prompt_embeddings=True)
    parser.add_argument(
        "--no_precompute_prompt_embeddings",
        dest="precompute_prompt_embeddings",
        action="store_false",
        help="debug fallback; formal runs precompute the four immutable text embeddings",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_steps != K41_SPACE.num_steps:
        raise SystemExit(
            f"this frozen runner requires --num_steps {K41_SPACE.num_steps}"
        )
    if args.chunk_size <= 0:
        raise SystemExit("--chunk_size must be positive")
    if args.conditioning_file is None and not args.allow_protocol_override:
        raise SystemExit(
            "formal exhaustive runs require --conditioning_file; runtime text "
            "encoding is allowed only with --allow_protocol_override"
        )
    if args.conditioning_file is not None and not args.conditioning_file.is_file():
        raise SystemExit(f"conditioning artifact not found: {args.conditioning_file}")
    if args.conditioning_file is not None and not args.precompute_prompt_embeddings:
        raise SystemExit(
            "--conditioning_file and --no_precompute_prompt_embeddings conflict"
        )

    frozen = {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_name": "flux-dev",
        "width": 1024,
        "height": 1024,
        "guidance": 3.5,
        "dtype": "bf16",
        "revision": "3de623fc",
        "seed": 42,
        "prompt_indices": "5,8,9,15",
    }
    overrides = {
        name: (getattr(args, name), expected)
        for name, expected in frozen.items()
        if getattr(args, name) != expected
    }
    if overrides and not args.allow_protocol_override:
        details = ", ".join(
            f"{name}={actual!r} (frozen {expected!r})"
            for name, (actual, expected) in overrides.items()
        )
        raise SystemExit(
            f"formal exhaustive protocol override rejected: {details}; "
            "use --allow_protocol_override only for smoke data"
        )

    prompts = read_prompts(args.prompt_file)
    prompt_indices = _parse_indices(args.prompt_indices, len(prompts))
    pairs = tuple((idx, prompts[idx], int(args.seed) + idx) for idx in prompt_indices)

    requested_start = int(args.rank_start)
    requested_end = K41_SPACE.total if args.rank_end is None else int(args.rank_end)
    if not (0 <= requested_start <= requested_end <= K41_SPACE.total):
        raise SystemExit(
            f"invalid requested interval [{requested_start}, {requested_end}) for "
            f"total {K41_SPACE.total}"
        )
    rank_start, rank_end = split_rank_interval(
        requested_start, requested_end, args.shard_count, args.shard_idx
    )
    if rank_start >= rank_end:
        print(
            f"[exhaustive-k41] shard {args.shard_idx}/{args.shard_count} has no ranks "
            f"inside requested interval [{requested_start}, {requested_end})",
            flush=True,
        )
        return 0

    if args.revision == "":
        args.revision = None
    args.conditioning_artifact_sha256 = (
        _file_sha256(args.conditioning_file)
        if args.conditioning_file is not None
        else None
    )
    experiment = _experiment_payload(
        args, prompt_indices, tuple(prompt for _idx, prompt, _seed in pairs)
    )
    experiment_fingerprint = _fingerprint(experiment)
    parts_dir = args.output_dir / "parts"
    workers_dir = args.output_dir / "workers"
    parts_dir.mkdir(parents=True, exist_ok=True)
    workers_dir.mkdir(parents=True, exist_ok=True)
    done_path = workers_dir / (
        f"worker_{args.shard_idx:05d}_of_{args.shard_count:05d}_"
        f"{rank_start:07d}_{rank_end:07d}.done.json"
    )
    if args.resume and done_path.is_file():
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"worker completion file is unreadable: {done_path}"
            ) from exc
        if (
            done.get("schema") != WORKER_SCHEMA
            or done.get("experiment_fingerprint") != experiment_fingerprint
            or (done.get("rank_start"), done.get("rank_end")) != (rank_start, rank_end)
        ):
            raise RuntimeError(
                f"worker completion file has wrong identity: {done_path}"
            )
        print(f"[exhaustive-k41] resume skip complete worker: {done_path}", flush=True)
        return 0

    import torch

    from flux.sp_cross_runner import resolve_model_commit

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    from diffusers import DiffusionPipeline
    from flux.oracle_runner import (
        _decode_to_pil,
        _run_one_pipe_call,
        install_oracle,
        reset_oracle_state,
    )

    weights = resolve_model_commit(args.model_id, args.revision)
    print(
        f"[exhaustive-k41] ranks=[{rank_start},{rank_end}) "
        f"shard={args.shard_idx}/{args.shard_count} prompts={prompt_indices} "
        f"revision={args.revision or '(hub default)'} "
        f"commit={weights['model_commit'] or 'UNRESOLVED'}",
        flush=True,
    )
    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, revision=args.revision
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start
    if weights.get("model_commit") is None:
        for config in (
            getattr(pipe, "config", None),
            getattr(pipe.transformer, "config", None),
        ):
            commit = getattr(config, "_commit_hash", None)
            if (
                isinstance(commit, str)
                and len(commit) == 40
                and not set(commit) - set("0123456789abcdef")
            ):
                weights["model_commit"] = commit
                weights["model_commit_source"] = "loaded_pipeline_config"
                break

    if args.conditioning_file is not None:
        conditioning = _load_conditioning_artifact(
            args.conditioning_file,
            pipe=pipe,
            args=args,
            prompt_indices=prompt_indices,
            selected_prompts=tuple(prompt for _idx, prompt, _seed in pairs),
            resolved_model_commit=weights.get("model_commit"),
        )
    elif args.precompute_prompt_embeddings:
        conditioning = {
            prompt_idx: _encode_prompt(pipe, prompt, args)
            for prompt_idx, prompt, _seed in pairs
        }
    else:
        conditioning = {}

    def run_one(prompt_idx: int, prompt: str, seed: int):
        if args.precompute_prompt_embeddings:
            return _run_conditioned(pipe, conditioning[prompt_idx], seed, args)
        return _run_one_pipe_call(pipe, prompt, seed, args)

    # One installed reuse path serves both full references (empty cache set)
    # and every enumerated schedule.  Only cache_steps_set and trajectory state
    # change between calls.
    teardown = install_oracle(
        pipe, cache_steps=(), num_steps=args.num_steps, cache_mode="seacache"
    )
    height = (args.height // 16) * 16
    width = (args.width // 16) * 16
    references: dict[int, np.ndarray] = {}
    worker_start = time.perf_counter()
    try:
        pipe.transformer.cache_steps_set = frozenset()
        for prompt_idx, prompt, seed in pairs:
            reset_oracle_state(pipe)
            latent = run_one(prompt_idx, prompt, seed)
            references[prompt_idx] = _image_array(
                _decode_to_pil(pipe, latent, height, width)
            )
            del latent

        chunk_start = rank_start
        while chunk_start < rank_end:
            chunk_end = min(chunk_start + args.chunk_size, rank_end)
            part_path = parts_dir / f"part_{chunk_start:07d}_{chunk_end:07d}.json"
            if part_path.exists():
                if not args.resume:
                    raise RuntimeError(
                        f"part already exists; pass --resume after checking identity: {part_path}"
                    )
                validate_existing_part(
                    part_path,
                    rank_start=chunk_start,
                    rank_end=chunk_end,
                    experiment_fingerprint=experiment_fingerprint,
                )
                print(f"[exhaustive-k41] resume skip {part_path.name}", flush=True)
                chunk_start = chunk_end
                continue

            rows: list[dict[str, Any]] = []
            for offset, variable_indices in enumerate(
                iter_lex_range(
                    chunk_start,
                    chunk_end,
                    len(K41_SPACE.variable_steps),
                    K41_SPACE.variable_full_count,
                )
            ):
                rank = chunk_start + offset
                full_steps, cache_steps, bits = schedule_for_rank(rank)
                expected_variable = tuple(
                    K41_SPACE.variable_steps[index] for index in variable_indices
                )
                if (
                    tuple(
                        step
                        for step in full_steps
                        if step not in K41_SPACE.forced_full_steps
                    )
                    != expected_variable
                ):
                    raise AssertionError(
                        "enumeration and schedule construction disagree"
                    )
                pipe.transformer.cache_steps_set = frozenset(cache_steps)
                per_prompt: list[dict[str, Any]] = []
                schedule_start = time.perf_counter()
                for prompt_idx, prompt, seed in pairs:
                    reset_oracle_state(pipe)
                    latent = run_one(prompt_idx, prompt, seed)
                    candidate = _image_array(
                        _decode_to_pil(pipe, latent, height, width)
                    )
                    mse, psnr = mse_and_psnr(references[prompt_idx], candidate)
                    per_prompt.append(
                        {
                            "prompt_idx": prompt_idx,
                            "seed": seed,
                            "mse": mse,
                            "psnr_db": psnr,
                            "zero_mse": mse == 0.0,
                        }
                    )
                    del latent, candidate
                psnrs = [record["psnr_db"] for record in per_prompt]
                rows.append(
                    {
                        "rank": rank,
                        "variable_full_steps": list(expected_variable),
                        "cache_mask_hex": mask_hex(bits),
                        "bits": bits,
                        "per_prompt": per_prompt,
                        "mean_psnr_db": float(sum(psnrs) / len(psnrs)),
                        "min_psnr_db": float(min(psnrs)),
                        "wall_s": float(time.perf_counter() - schedule_start),
                    }
                )

            part = {
                "schema": PART_SCHEMA,
                "experiment_fingerprint": experiment_fingerprint,
                "experiment": experiment,
                "space": K41_SPACE.identity_payload,
                "weights": weights,
                "rank_start": chunk_start,
                "rank_end": chunk_end,
                "rows": rows,
            }
            _atomic_write_json(part_path, part)
            elapsed = time.perf_counter() - worker_start
            completed = chunk_end - rank_start
            rate = completed / elapsed if elapsed else 0.0
            print(
                f"[exhaustive-k41] wrote {part_path.name}; "
                f"{completed}/{rank_end-rank_start} schedules, {rate:.3f} schedule/s",
                flush=True,
            )
            chunk_start = chunk_end
    finally:
        teardown()

    worker_done = {
        "schema": WORKER_SCHEMA,
        "experiment_fingerprint": experiment_fingerprint,
        "experiment": experiment,
        "weights": weights,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "requested_rank_start": requested_start,
        "requested_rank_end": requested_end,
        "rank_start": rank_start,
        "rank_end": rank_end,
        "model_load_s": float(model_load_s),
        "worker_wall_s": float(time.perf_counter() - worker_start),
        "device": torch.cuda.get_device_name(0),
    }
    _atomic_write_json(done_path, worker_done)
    print(f"[exhaustive-k41] complete: {done_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
