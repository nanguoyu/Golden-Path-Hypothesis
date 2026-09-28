#!/usr/bin/env python3
"""Long-lived Qwen-Image worker for Stage 5A fixed-schedule screening."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qwen_image._helpers import (  # noqa: E402
    atomic_write_json,
    git_sha,
    git_status_short,
    pipeline_identity,
)
from qwen_image.coarse_cache import (  # noqa: E402
    QwenCoarseConfig,
    install_qwen_coarse_forward,
    qwen_coarse_decisions,
    reset_qwen_coarse_state,
    restore_qwen_coarse_forward,
)
from qwen_image.runner import _save_image  # noqa: E402


CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
EXPECTED_PROMPT_INDICES = (0, 67, 133)
EXPECTED_BASE_SEEDS = (300042, 400042, 500042)
CELL_FIELDS = (
    "candidate_id",
    "ratio",
    "candidate_pool",
    "cell_idx",
    "prompt_idx",
    "prompt",
    "base_seed",
    "seed",
    "image_path",
    "decisions_path",
    "status",
    "generation_s",
    "n_cached",
    "model_id",
    "dtype",
    "num_steps",
    "width",
    "height",
    "true_cfg_scale",
    "negative_prompt_repr",
    "payload_mode",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    ratio: str
    candidate_pool: str
    bitstring: str

    @property
    def n_cached(self) -> int:
        return self.bitstring.count("1")


@dataclass(frozen=True)
class SearchCell:
    cell_idx: int
    prompt_idx: int
    prompt: str
    base_seed: int
    seed: int


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def load_candidates(path: Path, *, num_steps: int = 50) -> list[Candidate]:
    fields, rows = _read_tsv(path)
    required = {"candidate_id", "ratio", "candidate_pool", "bitstring", "k"}
    if missing := required.difference(fields):
        raise ValueError(f"{path}: missing candidate columns {sorted(missing)}")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"].strip()
        ratio = f"{float(row['ratio']):.2f}"
        pool = row["candidate_pool"].strip()
        bits = row["bitstring"].strip()
        if not CANDIDATE_ID_RE.fullmatch(candidate_id):
            raise ValueError(
                f"{path}:{line_number}: unsafe candidate_id {candidate_id!r}"
            )
        if candidate_id in seen:
            raise ValueError(
                f"{path}:{line_number}: duplicate candidate_id {candidate_id!r}"
            )
        if pool not in {"exact", "matched_control"}:
            raise ValueError(
                f"{path}:{line_number}: unsupported candidate_pool {pool!r}"
            )
        if len(bits) != num_steps or set(bits) - {"0", "1"}:
            raise ValueError(f"{path}:{line_number}: invalid {num_steps}-bit schedule")
        if int(row["k"]) != bits.count("1"):
            raise ValueError(f"{path}:{line_number}: K does not match bitstring")
        if bits[0] != "0" or bits[-1] != "0":
            raise ValueError(
                f"{path}:{line_number}: steps 0 and {num_steps - 1} must be full"
            )
        candidates.append(Candidate(candidate_id, ratio, pool, bits))
        seen.add(candidate_id)
    if not candidates:
        raise ValueError(f"candidate TSV is empty: {path}")
    if len({candidate.ratio for candidate in candidates}) != 1:
        raise ValueError(
            f"{path}: one worker shard must contain a single cache-ratio tier"
        )
    if len({candidate.n_cached for candidate in candidates}) != 1:
        raise ValueError(f"{path}: one worker shard must be fixed-budget")
    return candidates


def load_search_cells(path: Path) -> list[SearchCell]:
    fields, rows = _read_tsv(path)
    required = {"cell_idx", "prompt_idx", "prompt", "base_seed", "seed"}
    if missing := required.difference(fields):
        raise ValueError(f"{path}: missing search-cell columns {sorted(missing)}")
    cells: list[SearchCell] = []
    for row in rows:
        cell = SearchCell(
            cell_idx=int(row["cell_idx"]),
            prompt_idx=int(row["prompt_idx"]),
            prompt=row["prompt"],
            base_seed=int(row["base_seed"]),
            seed=int(row["seed"]),
        )
        if cell.seed != cell.base_seed + cell.prompt_idx:
            raise ValueError(f"{path}: seed rule mismatch in cell {cell.cell_idx}")
        cells.append(cell)
    if [cell.cell_idx for cell in cells] != list(range(9)):
        raise ValueError(f"{path}: Stage 5A requires contiguous cell_idx 0..8")
    observed_pairs = {(cell.prompt_idx, cell.base_seed) for cell in cells}
    expected_pairs = {
        (prompt_idx, base_seed)
        for prompt_idx in EXPECTED_PROMPT_INDICES
        for base_seed in EXPECTED_BASE_SEEDS
    }
    if observed_pairs != expected_pairs:
        raise ValueError(
            f"{path}: Stage 5A prompt/base-seed Cartesian cells differ from plan"
        )
    return cells


def _load_pipeline(model_id: str, *, torch_dtype: Any, device: str) -> Any:
    try:
        from diffusers import QwenImagePipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("diffusers.QwenImagePipeline is unavailable") from exc
    return QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(
        device
    )


def _image_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def _observed_bitstring(decisions: Mapping[str, Any], *, num_steps: int) -> str:
    rows = decisions.get("steps")
    if not isinstance(rows, list) or len(rows) != num_steps:
        raise RuntimeError(f"decision row count is not {num_steps}")
    bits: list[str] = []
    for expected_step, row in enumerate(rows):
        if not isinstance(row, Mapping) or int(row.get("step", -1)) != expected_step:
            raise RuntimeError("decision steps are not contiguous")
        action = row.get("action")
        if action not in {"full", "cache"}:
            raise RuntimeError(
                f"invalid decision action at step {expected_step}: {action!r}"
            )
        bits.append("1" if action == "cache" else "0")
    return "".join(bits)


def _existing_decision_matches(
    path: Path, candidate: Candidate, *, num_steps: int
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _observed_bitstring(payload, num_steps=num_steps) == candidate.bitstring
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return False


def _record_paths(
    output_dir: Path,
    *,
    mode: str,
    candidate_tsv: Path | None,
) -> tuple[Path, Path]:
    tag = "original_reference" if mode == "original" else Path(candidate_tsv).stem
    return output_dir / f"cells_{tag}.csv", output_dir / f"metadata_{tag}.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cached", "original"), default="cached")
    parser.add_argument("--cells-tsv", type=Path, required=True)
    parser.add_argument("--candidate-tsv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen-Image")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "cached" and args.candidate_tsv is None:
        parser.error("--candidate-tsv is required in cached mode")
    if args.mode == "original" and args.candidate_tsv is not None:
        parser.error("--candidate-tsv is not used in original mode")
    if args.num_steps != 50:
        parser.error("Stage 5A is frozen to 50 denoising steps")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cells = load_search_cells(args.cells_tsv)
    candidates = (
        load_candidates(args.candidate_tsv, num_steps=args.num_steps)
        if args.mode == "cached"
        else [Candidate("original", "-", "original", "0" * args.num_steps)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells_csv, metadata_path = _record_paths(
        args.output_dir,
        mode=args.mode,
        candidate_tsv=args.candidate_tsv,
    )

    import torch

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_started = time.perf_counter()
    pipe = _load_pipeline(
        args.model_id, torch_dtype=dtype_map[args.dtype], device=device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - process_started
    if args.mode == "cached":
        install_qwen_coarse_forward(
            pipe,
            QwenCoarseConfig(
                mode="SeaCachePayload",
                num_steps=args.num_steps,
                first_enhance=1,
                payload_mode="reuse",
                payload_schedule_dir=None,
                true_cfg=True,
            ),
        )
    else:
        restore_qwen_coarse_forward(pipe)

    generated = 0
    resumed = 0
    processed = 0
    generation_times: list[float] = []
    try:
        with cells_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CELL_FIELDS)
            writer.writeheader()
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_dir = args.output_dir / candidate.candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                for cell in cells:
                    image_path = candidate_dir / f"img_{cell.cell_idx}.png"
                    decision_path = (
                        candidate_dir / f"decisions_{cell.cell_idx:05d}.json"
                    )
                    complete = _image_complete(image_path)
                    if args.mode == "cached":
                        complete = complete and _existing_decision_matches(
                            decision_path,
                            candidate,
                            num_steps=args.num_steps,
                        )
                    if args.resume and complete:
                        status = "resumed"
                        generation_s: float | str = ""
                        resumed += 1
                    else:
                        if args.mode == "cached":
                            reset_qwen_coarse_state(
                                pipe,
                                prompt_idx=cell.prompt_idx,
                                seed=cell.seed,
                                locked_action_bitstring=candidate.bitstring,
                            )
                        generator = torch.Generator(device=device).manual_seed(
                            cell.seed
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        started = time.perf_counter()
                        result = pipe(
                            prompt=cell.prompt,
                            negative_prompt=args.negative_prompt,
                            true_cfg_scale=float(args.true_cfg_scale),
                            height=int(args.height),
                            width=int(args.width),
                            num_inference_steps=int(args.num_steps),
                            generator=generator,
                            output_type="pil",
                            return_dict=True,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        generation_s = time.perf_counter() - started
                        images = getattr(result, "images", None)
                        if not images:
                            raise RuntimeError(
                                f"Qwen-Image returned no image for {candidate.candidate_id}/{cell.cell_idx}"
                            )
                        _save_image(images[0], image_path, "png")
                        if args.mode == "cached":
                            decisions = qwen_coarse_decisions(pipe)
                            observed = _observed_bitstring(
                                decisions, num_steps=args.num_steps
                            )
                            if observed != candidate.bitstring:
                                raise RuntimeError(
                                    f"{candidate.candidate_id}/{cell.cell_idx} executed {observed}, "
                                    f"expected {candidate.bitstring}"
                                )
                            decisions.update(
                                {
                                    "candidate_id": candidate.candidate_id,
                                    "candidate_pool": candidate.candidate_pool,
                                    "ratio": candidate.ratio,
                                    "action_bitstring": candidate.bitstring,
                                    "cell_idx": cell.cell_idx,
                                    "prompt": cell.prompt,
                                    "base_seed": cell.base_seed,
                                    "image_file": image_path.name,
                                }
                            )
                            atomic_write_json(decision_path, decisions)
                        status = "generated"
                        generated += 1
                        generation_times.append(float(generation_s))

                    writer.writerow(
                        {
                            "candidate_id": candidate.candidate_id,
                            "ratio": candidate.ratio,
                            "candidate_pool": candidate.candidate_pool,
                            "cell_idx": cell.cell_idx,
                            "prompt_idx": cell.prompt_idx,
                            "prompt": cell.prompt,
                            "base_seed": cell.base_seed,
                            "seed": cell.seed,
                            "image_path": str(image_path.relative_to(args.output_dir)),
                            "decisions_path": (
                                str(decision_path.relative_to(args.output_dir))
                                if args.mode == "cached"
                                else "-"
                            ),
                            "status": status,
                            "generation_s": generation_s,
                            "n_cached": (
                                candidate.n_cached if args.mode == "cached" else 0
                            ),
                            "model_id": args.model_id,
                            "dtype": args.dtype,
                            "num_steps": args.num_steps,
                            "width": args.width,
                            "height": args.height,
                            "true_cfg_scale": args.true_cfg_scale,
                            "negative_prompt_repr": repr(args.negative_prompt),
                            "payload_mode": (
                                "reuse" if args.mode == "cached" else "original"
                            ),
                        }
                    )
                    processed += 1
                handle.flush()
                print(
                    f"[qwen-stage5a] candidate {candidate_index}/{len(candidates)} "
                    f"{candidate.candidate_id} complete",
                    flush=True,
                )
    finally:
        restore_qwen_coarse_forward(pipe)

    wallclock_s = time.perf_counter() - process_started
    metadata = {
        "schema": "qwen_stage5_search_worker.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(REPO),
        "git_status_short": git_status_short(REPO),
        "mode": args.mode,
        "candidate_tsv": str(args.candidate_tsv) if args.candidate_tsv else None,
        "cells_tsv": str(args.cells_tsv),
        "output_dir": str(args.output_dir),
        "model_id": args.model_id,
        "pipeline": pipeline_identity(pipe),
        "dtype": args.dtype,
        "num_steps": args.num_steps,
        "width": args.width,
        "height": args.height,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt_repr": repr(args.negative_prompt),
        "payload_mode": "reuse" if args.mode == "cached" else "original",
        "candidate_count": len(candidates),
        "cells_per_candidate": len(cells),
        "processed_cells": processed,
        "generated_cells": generated,
        "resumed_cells": resumed,
        "model_load_s": model_load_s,
        "mean_generated_cell_s": (
            sum(generation_times) / len(generation_times) if generation_times else None
        ),
        "wallclock_s": wallclock_s,
    }
    if not math.isfinite(wallclock_s) or processed != len(candidates) * len(cells):
        raise RuntimeError(
            "Stage 5A worker did not complete its assigned Cartesian cells"
        )
    atomic_write_json(metadata_path, metadata)
    print(
        f"[qwen-stage5a] done generated={generated} resumed={resumed} "
        f"cells={processed} wall_s={wallclock_s:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
