#!/usr/bin/env python3
"""Long-lived FLUX worker for Golden Path P6 generation.

Cached mode installs the existing SeaCache payload forward and evaluates every
candidate in one TSV shard. Original mode leaves the pipeline forward untouched
and generates one paired reference set. Both run the full prompt x seed
Cartesian product after loading FLUX once.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.io_utils import image_filename, read_prompts  # noqa: E402


_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_CELL_FIELDS = (
    "candidate_id",
    "cell_idx",
    "prompt_idx",
    "prompt",
    "seed",
    "image_path",
    "status",
    "generation_s",
    "n_cached",
    "model_id",
    "model_name",
    "dtype",
    "num_steps",
    "width",
    "height",
    "guidance",
    "payload_mode",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cache_steps: tuple[int, ...]


@dataclass(frozen=True)
class GenerationCell:
    candidate_id: str
    cache_steps: tuple[int, ...]
    prompt_idx: int
    prompt: str
    seed: int
    cell_idx: int


def _parse_step_list(raw: str) -> tuple[int, ...]:
    value = raw.strip()
    if not value:
        raise ValueError("cache-step field is empty; use [] for an empty schedule")
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("cache-step JSON must be a list")
        values = parsed
    else:
        values = [part.strip() for part in value.split(",") if part.strip()]
    try:
        steps = tuple(int(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid cache-step list: {raw!r}") from exc
    return steps


def _parse_bitstring(raw: str, *, num_steps: int) -> tuple[int, ...]:
    bits = raw.strip()
    if len(bits) != num_steps or set(bits) - {"0", "1"}:
        raise ValueError(
            f"bitstring must contain exactly {num_steps} zero/one characters"
        )
    return tuple(index for index, bit in enumerate(bits) if bit == "1")


def _validate_steps(
    steps: Iterable[int], *, num_steps: int, first_enhance: int
) -> tuple[int, ...]:
    canonical = tuple(int(step) for step in steps)
    if canonical != tuple(sorted(set(canonical))):
        raise ValueError("cache steps must be sorted and unique")
    if any(step < 0 or step >= num_steps for step in canonical):
        raise ValueError(f"cache steps must lie in [0, {num_steps})")

    forced_full = set(range(max(1, first_enhance))) | {num_steps - 1}
    forbidden = sorted(forced_full.intersection(canonical))
    if forbidden:
        raise ValueError(
            "fixed schedule requests cache on forced-full steps: "
            + ",".join(str(step) for step in forbidden)
        )
    return canonical


def _steps_from_row(
    row: dict[str, str], *, num_steps: int, first_enhance: int
) -> tuple[int, ...]:
    representations: list[tuple[str, tuple[int, ...]]] = []
    for column in ("cache_steps", "cached_steps"):
        raw = (row.get(column) or "").strip()
        if raw:
            representations.append((column, _parse_step_list(raw)))
    raw_bits = (row.get("bitstring") or "").strip()
    if raw_bits:
        representations.append(
            ("bitstring", _parse_bitstring(raw_bits, num_steps=num_steps))
        )
    if not representations:
        raise ValueError(
            "candidate row needs cache_steps, cached_steps, or bitstring"
        )

    steps = representations[0][1]
    for name, other in representations[1:]:
        if other != steps:
            raise ValueError(
                f"candidate schedule representations disagree at {name}"
            )
    return _validate_steps(
        steps, num_steps=num_steps, first_enhance=first_enhance
    )


def load_candidates(
    path: Path, *, num_steps: int, first_enhance: int
) -> list[Candidate]:
    """Read one candidate TSV shard and validate fixed-budget schedules."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if first_enhance < 0 or first_enhance >= num_steps:
        raise ValueError("first_enhance must lie in [0, num_steps)")

    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "candidate_id" not in reader.fieldnames:
            raise ValueError("candidate TSV must contain a candidate_id column")
        for line_number, row in enumerate(reader, start=2):
            candidate_id = (row.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
                raise ValueError(
                    f"line {line_number}: unsafe candidate_id {candidate_id!r}"
                )
            if candidate_id in seen_ids:
                raise ValueError(
                    f"line {line_number}: duplicate candidate_id {candidate_id!r}"
                )
            try:
                cache_steps = _steps_from_row(
                    row, num_steps=num_steps, first_enhance=first_enhance
                )
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number}, candidate {candidate_id!r}: {exc}"
                ) from exc

            raw_n_cached = (row.get("n_cached") or "").strip()
            if raw_n_cached and int(raw_n_cached) != len(cache_steps):
                raise ValueError(
                    f"line {line_number}: n_cached does not match schedule"
                )
            seen_ids.add(candidate_id)
            candidates.append(Candidate(candidate_id, cache_steps))

    if not candidates:
        raise ValueError(f"candidate TSV has no candidates: {path}")
    budgets = {len(candidate.cache_steps) for candidate in candidates}
    if len(budgets) != 1:
        raise ValueError(
            f"candidate TSV is not fixed-budget: cache counts={sorted(budgets)}"
        )
    return candidates


def normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("at least one seed is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique")
    return normalized


def expand_cells(
    candidates: Sequence[Candidate], prompts: Sequence[str], seeds: Sequence[int]
) -> list[GenerationCell]:
    """Expand candidates into a true prompt x seed Cartesian product."""
    seed_values = normalize_seeds(seeds)
    if not prompts:
        raise ValueError("at least one prompt is required")
    cells: list[GenerationCell] = []
    for candidate in candidates:
        for prompt_idx, prompt in enumerate(prompts):
            for seed_idx, seed in enumerate(seed_values):
                cells.append(
                    GenerationCell(
                        candidate_id=candidate.candidate_id,
                        cache_steps=candidate.cache_steps,
                        prompt_idx=prompt_idx,
                        prompt=str(prompt),
                        seed=seed,
                        cell_idx=prompt_idx * len(seed_values) + seed_idx,
                    )
                )
    return cells


def cell_image_path(output_dir: Path, cell: GenerationCell) -> Path:
    return Path(output_dir) / cell.candidate_id / image_filename(cell.cell_idx)


def output_record_paths(
    output_dir: Path, *, mode: str, candidate_tsv: Path | None
) -> tuple[Path, Path]:
    if mode == "original":
        tag = "original_reference"
    else:
        if candidate_tsv is None:
            raise ValueError("cached mode requires candidate_tsv")
        tag = candidate_tsv.stem
    root = Path(output_dir)
    return root / f"cells_{tag}.csv", root / f"metadata_{tag}.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-lived FLUX generation worker for paired P6 cells"
    )
    parser.add_argument("--mode", choices=("cached", "original"), default="cached")
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument(
        "--candidate_tsv",
        type=Path,
        help="Fixed-schedule TSV shard; required only in cached mode.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev"
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--first_enhance", type=int, default=1)
    parser.add_argument("--seacache_thresh", type=float, default=0.0)
    parser.add_argument(
        "--payload_mode", choices=("taylor_o1", "reuse"), default="taylor_o1"
    )
    parser.add_argument("--payload_blend", type=float, default=1.0)
    parser.add_argument("--payload_sigma", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "cached" and args.candidate_tsv is None:
        parser.error("--candidate_tsv is required in cached mode")
    return args


def _observed_cache_steps(pipe, *, num_steps: int) -> tuple[int, ...]:
    decisions = list(getattr(pipe.transformer, "seacache_payload_decisions", []))
    if len(decisions) != num_steps:
        raise RuntimeError(
            f"payload decision count mismatch: got {len(decisions)}, expected {num_steps}"
        )
    return tuple(
        int(row["step"]) for row in decisions if int(row.get("u", 0)) == 1
    )


def main() -> int:
    args = _parse_args()
    seeds = normalize_seeds(args.seeds)
    if args.mode == "cached":
        candidates = load_candidates(
            args.candidate_tsv,
            num_steps=int(args.num_steps),
            first_enhance=int(args.first_enhance),
        )
    else:
        candidates = [Candidate("original", ())]
    prompts = read_prompts(args.prompt_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells_csv, metadata_path = output_record_paths(
        args.output_dir, mode=args.mode, candidate_tsv=args.candidate_tsv
    )

    import torch
    from diffusers import DiffusionPipeline

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    print(
        f"Loading {args.model_id} once in {args.mode} mode for "
        f"{len(candidates)} candidates, "
        f"{len(prompts)} prompts, seeds={list(seeds)}",
        flush=True,
    )
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]
    ).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - started

    seacache_payload = None
    teardown = None
    if args.mode == "cached":
        from flux import seacache_payload as seacache_payload_module

        seacache_payload = seacache_payload_module
        teardown = seacache_payload.install(
            pipe,
            threshold=float(args.seacache_thresh),
            num_steps=int(args.num_steps),
            first_enhance=int(args.first_enhance),
            payload_mode=str(args.payload_mode),
            payload_blend=float(args.payload_blend),
            payload_sigma=float(args.payload_sigma),
        )

    generated = 0
    resumed = 0
    processed = 0
    actual_width = (int(args.width) // 16) * 16
    actual_height = (int(args.height) // 16) * 16
    actual_guidance = 0.0 if args.model_name == "flux-schnell" else float(args.guidance)
    cells_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cells_csv.open("w", encoding="utf-8", newline="") as cells_handle:
            cells_writer = csv.DictWriter(cells_handle, fieldnames=_CELL_FIELDS)
            cells_writer.writeheader()
            for candidate_idx, candidate in enumerate(candidates, start=1):
                candidate_dir = args.output_dir / candidate.candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                action_steps = set(candidate.cache_steps)
                candidate_cells = expand_cells([candidate], prompts, seeds)
                for cell in candidate_cells:
                    image_path = cell_image_path(args.output_dir, cell)
                    relative_path = image_path.relative_to(args.output_dir)
                    if args.resume and image_path.is_file():
                        status = "resumed"
                        generation_s: object = ""
                        resumed += 1
                    else:
                        if seacache_payload is not None:
                            seacache_payload.reset_per_image_state(
                                pipe,
                                action_steps=action_steps,
                                prompt_idx=cell.cell_idx,
                            )
                        generator = torch.Generator(device=device).manual_seed(cell.seed)
                        cell_started = time.perf_counter()
                        result = pipe(
                            prompt=cell.prompt,
                            num_inference_steps=int(args.num_steps),
                            guidance_scale=actual_guidance,
                            height=actual_height,
                            width=actual_width,
                            max_sequence_length=(
                                256 if args.model_name == "flux-schnell" else 512
                            ),
                            num_images_per_prompt=1,
                            generator=generator,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        generation_s = time.perf_counter() - cell_started
                        if seacache_payload is not None:
                            observed = _observed_cache_steps(
                                pipe, num_steps=int(args.num_steps)
                            )
                            if observed != candidate.cache_steps:
                                raise RuntimeError(
                                    f"candidate {candidate.candidate_id} executed {observed}, "
                                    f"expected {candidate.cache_steps}"
                                )
                        result.images[0].save(image_path)
                        status = "generated"
                        generated += 1

                    cells_writer.writerow(
                        {
                            "candidate_id": candidate.candidate_id,
                            "cell_idx": cell.cell_idx,
                            "prompt_idx": cell.prompt_idx,
                            "prompt": cell.prompt,
                            "seed": cell.seed,
                            "image_path": str(relative_path),
                            "status": status,
                            "generation_s": generation_s,
                            "n_cached": len(candidate.cache_steps),
                            "model_id": str(args.model_id),
                            "model_name": str(args.model_name),
                            "dtype": str(args.dtype),
                            "num_steps": int(args.num_steps),
                            "width": actual_width,
                            "height": actual_height,
                            "guidance": actual_guidance,
                            "payload_mode": (
                                str(args.payload_mode)
                                if args.mode == "cached"
                                else "original"
                            ),
                        }
                    )
                    processed += 1
                cells_handle.flush()
                print(
                    f"candidate {candidate_idx}/{len(candidates)} "
                    f"{candidate.candidate_id} complete",
                    flush=True,
                )
    finally:
        if teardown is not None:
            teardown()

    wallclock_s = time.perf_counter() - started
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(args.mode),
        "prompt_file": str(args.prompt_file),
        "candidate_tsv": (
            str(args.candidate_tsv) if args.mode == "cached" else None
        ),
        "output_dir": str(args.output_dir),
        "model_id": str(args.model_id),
        "model_name": str(args.model_name),
        "dtype": str(args.dtype),
        "num_steps": int(args.num_steps),
        "width": actual_width,
        "height": actual_height,
        "guidance": actual_guidance,
        "first_enhance": int(args.first_enhance) if args.mode == "cached" else None,
        "seacache_thresh": (
            float(args.seacache_thresh) if args.mode == "cached" else None
        ),
        "payload_mode": str(args.payload_mode) if args.mode == "cached" else None,
        "payload_blend": float(args.payload_blend) if args.mode == "cached" else None,
        "payload_sigma": float(args.payload_sigma) if args.mode == "cached" else None,
        "seeds": list(seeds),
        "prompt_count": len(prompts),
        "candidate_count": len(candidates),
        "cells_per_candidate": len(prompts) * len(seeds),
        "generated_cells": generated,
        "resumed_cells": resumed,
        "model_load_s": model_load_s,
        "wallclock_s": wallclock_s,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "cache_steps": list(candidate.cache_steps),
            }
            for candidate in candidates
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"complete: generated={generated}, resumed={resumed}, "
        f"cells={processed}, wallclock={wallclock_s:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
