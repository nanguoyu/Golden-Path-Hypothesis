#!/usr/bin/env python3
"""Evaluate the frozen 337-schedule FLUX K41 family on held-out pairs.

One process owns a contiguous shard of prompt indices.  FLUX remains resident,
each prompt is encoded once, and the full reference plus every fixed residual-
reuse schedule use the same initial noise.  A completed prompt--seed pair is
committed as one atomic JSON file.  Only the full reference and the three
predeclared primary/control schedules are saved as PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.exhaustive_k41_runner import (  # noqa: E402
    _encode_prompt,
    _image_array,
    _run_conditioned,
    mse_and_psnr,
)
from lib.io_utils import image_filename, read_prompts, seed_for, split_shard  # noqa: E402


PAIR_SCHEMA = "flux_exhaustive_k41_family_pair.v1"
MODEL_COMMIT = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
EXPECTED_SOFTWARE = {
    "python": "3.12.13",
    "torch": "2.12.0",
    "diffusers": "0.38.0",
    "transformers": "5.8.1",
    "tokenizers": "0.22.2",
    "numpy": "2.4.4",
    "pillow": "10.4.0",
}
DEFAULT_MANIFEST = Path(
    "resources/exhaustive_k41/formal_results/candidate_manifest.tsv"
)
DEFAULT_SCHEDULE_DIR = Path(
    "resources/exhaustive_k41/formal_results/schedules"
)
DEFAULT_SAVED_RANKS = {
    "mean_optimal": 164762,
    "robust_optimal": 165962,
    "budcache": 176543,
}
EXPECTED_SAVED_FULL_STEPS = {
    "mean_optimal": (0, 1, 2, 4, 6, 11, 24, 41, 49),
    "robust_optimal": (0, 1, 2, 4, 6, 13, 23, 40, 49),
    "budcache": (0, 1, 2, 4, 7, 13, 20, 39, 49),
}
FORCED_FULL_STEPS = frozenset({0, 1, 2, 49})


@dataclass(frozen=True)
class Candidate:
    rank: int
    reasons: str
    variable_full_steps: tuple[int, ...]
    bits: str
    cache_steps: tuple[int, ...]

    @property
    def full_steps(self) -> tuple[int, ...]:
        return tuple(step for step, bit in enumerate(self.bits) if bit == "0")


def load_schedule_file(path: Path, *, num_steps: int) -> tuple[int, ...]:
    bits = path.read_text(encoding="utf-8").strip()
    if len(bits) != num_steps or set(bits) - {"0", "1"}:
        raise ValueError(f"{path} must contain exactly {num_steps} schedule bits")
    return tuple(step for step, bit in enumerate(bits) if bit == "1")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _atomic_save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    image.save(tmp)
    tmp.replace(path)


def _parse_int_set(value: str | None) -> frozenset[int]:
    if not value:
        return frozenset()
    try:
        result = frozenset(int(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise SystemExit(f"invalid comma-separated indices: {value!r}") from exc
    if any(index < 0 for index in result):
        raise SystemExit("indices must be non-negative")
    return result


def load_candidates(
    manifest: Path,
    schedule_dir: Path,
    *,
    num_steps: int = 50,
    expected_count: int = 337,
) -> list[Candidate]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != expected_count:
        raise ValueError(f"{manifest} has {len(rows)} rows; expected {expected_count}")

    candidates: list[Candidate] = []
    for row in rows:
        rank = int(row["rank"])
        schedule_file = schedule_dir / f"rank_{rank:07d}.txt"
        cache_steps = load_schedule_file(schedule_file, num_steps=num_steps)
        bits = schedule_file.read_text(encoding="utf-8").strip()
        variable = tuple(
            int(piece) for piece in row["variable_full_steps"].split(",") if piece
        )
        full_steps = frozenset(step for step, bit in enumerate(bits) if bit == "0")
        if len(cache_steps) != 41 or len(full_steps) != 9:
            raise ValueError(f"rank {rank} is not a K41 schedule")
        if not FORCED_FULL_STEPS.issubset(full_steps):
            raise ValueError(f"rank {rank} caches a forced-full step")
        if tuple(sorted(full_steps - FORCED_FULL_STEPS)) != variable:
            raise ValueError(f"rank {rank} disagrees with manifest variable_full_steps")
        candidates.append(
            Candidate(
                rank=rank,
                reasons=row["reasons"],
                variable_full_steps=variable,
                bits=bits,
                cache_steps=cache_steps,
            )
        )

    ranks = [candidate.rank for candidate in candidates]
    bits = [candidate.bits for candidate in candidates]
    if len(set(ranks)) != expected_count or len(set(bits)) != expected_count:
        raise ValueError("candidate ranks and schedules must both be unique")
    return candidates


def _pair_path(output_dir: Path, prompt_idx: int) -> Path:
    return output_dir / "pairs" / f"pair_{prompt_idx:05d}.json"


def _image_paths(
    output_dir: Path, prompt_idx: int, saved_ranks: dict[str, int]
) -> dict[str, Path]:
    filename = image_filename(prompt_idx)
    return {
        "reference": output_dir / "reference" / filename,
        **{label: output_dir / label / filename for label in saved_ranks},
    }


def validate_completed_pair(
    path: Path,
    *,
    prompt_idx: int,
    seed: int,
    protocol: dict[str, Any],
    candidate_ranks: list[int],
    image_paths: dict[str, Path],
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"completed pair is unreadable: {path}") from exc
    if payload.get("schema") != PAIR_SCHEMA:
        raise RuntimeError(f"completed pair has wrong schema: {path}")
    if (payload.get("prompt_idx"), payload.get("seed")) != (prompt_idx, seed):
        raise RuntimeError(f"completed pair has wrong prompt/seed identity: {path}")
    if payload.get("protocol") != protocol:
        raise RuntimeError(f"completed pair belongs to another protocol: {path}")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or [row.get("rank") for row in rows] != candidate_ranks:
        raise RuntimeError(f"completed pair has incomplete candidate rows: {path}")
    missing = [str(image) for image in image_paths.values() if not image.is_file()]
    if missing:
        raise RuntimeError(f"completed pair is missing saved images: {missing}")


def _software_versions() -> dict[str, str]:
    import diffusers
    import PIL
    import tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def validate_software_versions(actual: dict[str, str]) -> None:
    normalized = dict(actual)
    normalized["torch"] = normalized["torch"].split("+", 1)[0]
    mismatches = {
        name: {"actual": normalized.get(name), "expected": expected}
        for name, expected in EXPECTED_SOFTWARE.items()
        if normalized.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"formal family environment is not frozen: {mismatches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--exclude_indices", default="")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schedule_dir", type=Path, default=DEFAULT_SCHEDULE_DIR)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", default=MODEL_COMMIT)
    parser.add_argument("--model_name", choices=("flux-dev",), default="flux-dev")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_steps != 50 or args.revision != MODEL_COMMIT:
        raise SystemExit("formal family runs require 50 steps and the frozen FLUX commit")
    if args.width != 1024 or args.height != 1024 or args.guidance != 3.5:
        raise SystemExit("formal family runs require 1024x1024 and guidance 3.5")

    candidates = load_candidates(args.manifest, args.schedule_dir, num_steps=args.num_steps)
    candidate_ranks = [candidate.rank for candidate in candidates]
    saved_ranks = dict(DEFAULT_SAVED_RANKS)
    unknown_saved = sorted(set(saved_ranks.values()) - set(candidate_ranks))
    if unknown_saved:
        raise SystemExit(f"saved ranks are absent from the candidate family: {unknown_saved}")
    by_rank = {candidate.rank: candidate for candidate in candidates}
    for label, expected_steps in EXPECTED_SAVED_FULL_STEPS.items():
        if label in saved_ranks and by_rank[saved_ranks[label]].full_steps != expected_steps:
            raise SystemExit(f"saved rank {label} does not match its frozen full steps")

    prompts = read_prompts(args.prompt_file)
    excluded = _parse_int_set(args.exclude_indices)
    if excluded and max(excluded) >= len(prompts):
        raise SystemExit("--exclude_indices contains an index outside the prompt file")
    pairs = [(idx, prompt) for idx, prompt in enumerate(prompts) if idx not in excluded]
    lo, hi = split_shard(len(pairs), args.shard_count, args.shard_idx)
    selected = pairs[lo:hi]
    if not selected:
        print(f"[k41-family] shard {args.shard_idx}/{args.shard_count} is empty")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    software = _software_versions()
    validate_software_versions(software)
    protocol = {
        "model_id": args.model_id,
        "model_commit": MODEL_COMMIT,
        "num_steps": 50,
        "width": 1024,
        "height": 1024,
        "guidance": 3.5,
        "dtype": "bf16",
        "cache_count": 41,
        "payload": "residual_reuse",
        "prompt_file": str(args.prompt_file),
        "base_seed": int(args.base_seed),
        "seed_rule": "base_seed_plus_prompt_index",
        "excluded_indices": sorted(excluded),
        "candidate_manifest": str(args.manifest),
        "candidate_count": len(candidates),
        "saved_ranks": saved_ranks,
        "software": software,
    }
    pending: list[tuple[int, str]] = []
    skipped = 0
    for prompt_idx, prompt in selected:
        seed = seed_for(args.base_seed, prompt_idx)
        pair_path = _pair_path(args.output_dir, prompt_idx)
        images = _image_paths(args.output_dir, prompt_idx, saved_ranks)
        if not pair_path.exists():
            pending.append((prompt_idx, prompt))
            continue
        if not args.resume:
            raise RuntimeError(f"pair already exists; pass --resume: {pair_path}")
        validate_completed_pair(
            pair_path,
            prompt_idx=prompt_idx,
            seed=seed,
            protocol=protocol,
            candidate_ranks=candidate_ranks,
            image_paths=images,
        )
        skipped += 1
        print(f"[k41-family] resume skip idx={prompt_idx}", flush=True)
    if not pending:
        print(
            f"[k41-family] shard {args.shard_idx}/{args.shard_count} already complete",
            flush=True,
        )
        return 0

    import torch
    from diffusers import DiffusionPipeline
    from flux.oracle_runner import _decode_to_pil, install_oracle, reset_oracle_state
    from flux.sp_cross_runner import resolve_model_commit

    dtype = torch.bfloat16
    weights = resolve_model_commit(args.model_id, args.revision)
    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, revision=args.revision
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start
    if weights.get("model_commit") is None:
        for config in (getattr(pipe, "config", None), getattr(pipe.transformer, "config", None)):
            commit = getattr(config, "_commit_hash", None)
            if isinstance(commit, str) and len(commit) == 40:
                weights["model_commit"] = commit
                weights["model_commit_source"] = "loaded_pipeline_config"
                break
    if weights.get("model_commit") != MODEL_COMMIT:
        raise RuntimeError(
            f"loaded model commit {weights.get('model_commit')!r}; expected {MODEL_COMMIT}"
        )

    teardown = install_oracle(pipe, cache_steps=(), num_steps=50, cache_mode="seacache")
    height = (args.height // 16) * 16
    width = (args.width // 16) * 16
    shard_start = time.perf_counter()
    completed = 0
    try:
        for prompt_idx, prompt in pending:
            seed = seed_for(args.base_seed, prompt_idx)
            pair_path = _pair_path(args.output_dir, prompt_idx)
            images = _image_paths(args.output_dir, prompt_idx, saved_ranks)
            conditioning = _encode_prompt(pipe, prompt, args)
            pair_start = time.perf_counter()
            pipe.transformer.cache_steps_set = frozenset()
            reset_oracle_state(pipe)
            reference_start = time.perf_counter()
            latent = _run_conditioned(pipe, conditioning, seed, args)
            reference_image = _decode_to_pil(pipe, latent, height, width)
            torch.cuda.synchronize()
            reference_s = time.perf_counter() - reference_start
            reference_array = _image_array(reference_image)
            _atomic_save_image(reference_image, images["reference"])
            del latent, reference_image

            rows: list[dict[str, Any]] = []
            label_by_rank = {rank: label for label, rank in saved_ranks.items()}
            for candidate in candidates:
                pipe.transformer.cache_steps_set = frozenset(candidate.cache_steps)
                reset_oracle_state(pipe)
                generation_start = time.perf_counter()
                latent = _run_conditioned(pipe, conditioning, seed, args)
                image = _decode_to_pil(pipe, latent, height, width)
                torch.cuda.synchronize()
                generation_s = time.perf_counter() - generation_start
                array = _image_array(image)
                mse, psnr = mse_and_psnr(reference_array, array)
                label = label_by_rank.get(candidate.rank)
                if label is not None:
                    _atomic_save_image(image, images[label])
                rows.append(
                    {
                        "rank": candidate.rank,
                        "mse": mse,
                        "psnr_db": psnr,
                        "generation_s": generation_s,
                    }
                )
                del latent, image, array

            payload = {
                "schema": PAIR_SCHEMA,
                "protocol": protocol,
                "prompt_idx": prompt_idx,
                "seed": seed,
                "prompt": prompt,
                "reference_image": str(images["reference"].relative_to(args.output_dir)),
                "saved_images": {
                    label: str(images[label].relative_to(args.output_dir))
                    for label in saved_ranks
                },
                "reference_generation_s": reference_s,
                "pair_wall_s": time.perf_counter() - pair_start,
                "candidates": rows,
            }
            _atomic_write_json(pair_path, payload)
            completed += 1
            print(
                f"[k41-family] idx={prompt_idx} seed={seed} "
                f"candidates={len(rows)} wall={payload['pair_wall_s']:.1f}s",
                flush=True,
            )
    finally:
        teardown()

    summary = {
        "schema": "flux_exhaustive_k41_family_shard.v1",
        "protocol": protocol,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "selected_pairs": len(selected),
        "completed_pairs": completed,
        "skipped_pairs": skipped,
        "model_load_s": model_load_s,
        "wall_s": time.perf_counter() - shard_start,
        "device": torch.cuda.get_device_name(0),
    }
    _atomic_write_json(
        args.output_dir
        / "shards"
        / f"shard_{args.shard_idx:05d}_of_{args.shard_count:05d}.json",
        summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
