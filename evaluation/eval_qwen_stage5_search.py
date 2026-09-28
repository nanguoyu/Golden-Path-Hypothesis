#!/usr/bin/env python3
"""Evaluate Qwen-Image Stage 5A candidate shards with the standard five metrics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


OUTPUT_FIELDS = (
    "ratio",
    "candidate_pool",
    "candidate_id",
    "bitstring",
    "n_cached",
    "hamming_to_primary",
    "cell_idx",
    "prompt_idx",
    "prompt",
    "base_seed",
    "seed",
    "candidate_image",
    "original_image",
    "psnr",
    "ssim",
    "lpips",
    "image_reward",
    "clip",
)
PAIR_PROTOCOL_FIELDS = (
    "prompt",
    "base_seed",
    "model_id",
    "dtype",
    "num_steps",
    "width",
    "height",
    "true_cfg_scale",
    "negative_prompt_repr",
)


@dataclass(frozen=True)
class CandidateInfo:
    candidate_id: str
    ratio: str
    candidate_pool: str
    bitstring: str
    hamming_to_primary: int


@dataclass(frozen=True)
class PairedCell:
    candidate: CandidateInfo
    cell_idx: int
    prompt_idx: int
    prompt: str
    base_seed: int
    seed: int
    candidate_image: Path
    original_image: Path


def _compute_psnr_ssim(entries):
    from evaluation.eval_metrics import compute_psnr_ssim

    return compute_psnr_ssim(entries, want_psnr=True, want_ssim=True)


def _compute_lpips(entries, device: str, net: str) -> list[float]:
    from evaluation.eval_metrics import compute_lpips

    return compute_lpips(entries, device, net)


def _compute_image_reward(entries, device: str, model_name: str) -> list[float]:
    from evaluation.eval_metrics import compute_image_reward

    return compute_image_reward(entries, device, model_name)


def _compute_clip(entries, device: str, model_name: str) -> list[float]:
    from evaluation.eval_metrics import compute_clip

    return compute_clip(entries, device, model_name)


def _read_table(
    path: Path, *, delimiter: str
) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"table has no header: {path}")
        return list(reader.fieldnames), list(reader)


def load_candidate_info(paths: Sequence[Path]) -> dict[str, CandidateInfo]:
    if not paths:
        raise ValueError("at least one candidate TSV is required")
    required = {
        "candidate_id",
        "ratio",
        "candidate_pool",
        "bitstring",
        "hamming_to_primary",
    }
    candidates: dict[str, CandidateInfo] = {}
    for path in paths:
        fields, rows = _read_table(path, delimiter="\t")
        if missing := required.difference(fields):
            raise ValueError(f"{path}: missing candidate columns {sorted(missing)}")
        for row in rows:
            candidate_id = row["candidate_id"].strip()
            info = CandidateInfo(
                candidate_id=candidate_id,
                ratio=f"{float(row['ratio']):.2f}",
                candidate_pool=row["candidate_pool"].strip(),
                bitstring=row["bitstring"].strip(),
                hamming_to_primary=int(row["hamming_to_primary"]),
            )
            if not candidate_id or info.candidate_pool not in {
                "exact",
                "matched_control",
            }:
                raise ValueError(
                    f"{path}: invalid candidate metadata for {candidate_id!r}"
                )
            if len(info.bitstring) != 50 or set(info.bitstring) - {"0", "1"}:
                raise ValueError(f"{path}: invalid bitstring for {candidate_id!r}")
            if candidate_id in candidates:
                raise ValueError(f"duplicate candidate_id {candidate_id!r}")
            candidates[candidate_id] = info
    if not candidates:
        raise ValueError("candidate TSVs contain no candidates")
    return candidates


def _cell_key(row: Mapping[str, str], *, source: Path) -> tuple[int, int]:
    try:
        return int(row["prompt_idx"]), int(row["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid prompt_idx/seed row") from exc


def _resolve_image(root: Path, row: Mapping[str, str], *, source: Path) -> Path:
    value = str(row.get("image_path", "")).strip()
    if not value:
        raise ValueError(f"{source}: cells row has no image_path")
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _pair_signature(row: Mapping[str, str], *, source: Path) -> tuple[object, ...]:
    try:
        return (
            row["prompt"],
            int(row["base_seed"]),
            row["model_id"],
            row["dtype"],
            int(row["num_steps"]),
            int(row["width"]),
            int(row["height"]),
            float(row["true_cfg_scale"]),
            row["negative_prompt_repr"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid paired-generation protocol row") from exc


def pair_cells(
    *,
    cells_paths: Sequence[Path],
    candidate_tsvs: Sequence[Path],
    candidate_root: Path,
    original_cells_path: Path,
    original_root: Path,
) -> list[PairedCell]:
    if len(cells_paths) != len(candidate_tsvs):
        raise ValueError("--cells and --candidate-tsv counts must match")
    candidates = load_candidate_info(candidate_tsvs)
    required_cells = {
        "candidate_id",
        "cell_idx",
        "prompt_idx",
        "seed",
        "image_path",
        *PAIR_PROTOCOL_FIELDS,
    }

    original_fields, original_rows = _read_table(original_cells_path, delimiter=",")
    if missing := required_cells.difference(original_fields):
        raise ValueError(
            f"{original_cells_path}: missing cells columns {sorted(missing)}"
        )
    originals: dict[tuple[int, int], tuple[Path, tuple[object, ...]]] = {}
    for row in original_rows:
        key = _cell_key(row, source=original_cells_path)
        if key in originals:
            raise ValueError(f"{original_cells_path}: duplicate original key {key}")
        image = _resolve_image(original_root, row, source=original_cells_path)
        if not image.is_file():
            raise FileNotFoundError(f"missing original image for {key}: {image}")
        originals[key] = (image, _pair_signature(row, source=original_cells_path))
    if len(originals) != 9:
        raise ValueError(
            f"{original_cells_path}: Stage 5A requires exactly nine original cells"
        )

    paired: list[PairedCell] = []
    seen: set[tuple[str, int, int]] = set()
    counts: dict[str, int] = {candidate_id: 0 for candidate_id in candidates}
    for cells_path, candidate_tsv in zip(cells_paths, candidate_tsvs):
        shard_ids = set(load_candidate_info([candidate_tsv]))
        fields, rows = _read_table(cells_path, delimiter=",")
        if missing := required_cells.difference(fields):
            raise ValueError(f"{cells_path}: missing cells columns {sorted(missing)}")
        for row in rows:
            candidate_id = row["candidate_id"].strip()
            if candidate_id not in shard_ids:
                raise ValueError(
                    f"{cells_path}: candidate {candidate_id!r} is not in its shard"
                )
            key = _cell_key(row, source=cells_path)
            unique_key = (candidate_id, *key)
            if unique_key in seen:
                raise ValueError(f"duplicate candidate/prompt/seed cell {unique_key}")
            if key not in originals:
                raise ValueError(f"{cells_path}: no original for prompt_idx/seed {key}")
            original_image, original_signature = originals[key]
            if _pair_signature(row, source=cells_path) != original_signature:
                raise ValueError(
                    f"{cells_path}: protocol mismatch for prompt_idx/seed {key}"
                )
            candidate_image = _resolve_image(candidate_root, row, source=cells_path)
            if not candidate_image.is_file():
                raise FileNotFoundError(
                    f"missing candidate image for {unique_key}: {candidate_image}"
                )
            paired.append(
                PairedCell(
                    candidate=candidates[candidate_id],
                    cell_idx=int(row["cell_idx"]),
                    prompt_idx=key[0],
                    prompt=row["prompt"],
                    base_seed=int(row["base_seed"]),
                    seed=key[1],
                    candidate_image=candidate_image,
                    original_image=original_image,
                )
            )
            counts[candidate_id] += 1
            seen.add(unique_key)
    bad_counts = {
        candidate_id: count for candidate_id, count in counts.items() if count != 9
    }
    if bad_counts:
        raise ValueError(f"each candidate must have nine paired cells: {bad_counts}")
    return sorted(paired, key=lambda cell: (cell.candidate.candidate_id, cell.cell_idx))


def evaluate_pairs(
    pairs: Sequence[PairedCell],
    *,
    device: str,
    lpips_net: str,
    image_reward_model: str,
    clip_model: str,
) -> list[dict[str, object]]:
    if not pairs:
        raise ValueError("no paired cells to evaluate")
    entries = [
        (index, cell.candidate_image, cell.original_image, cell.prompt)
        for index, cell in enumerate(pairs)
    ]
    classic = _compute_psnr_ssim(entries)
    metric_values = {
        "psnr": classic["psnr"],
        "ssim": classic["ssim"],
        "lpips": _compute_lpips(entries, device, lpips_net),
        "image_reward": _compute_image_reward(entries, device, image_reward_model),
        "clip": _compute_clip(entries, device, clip_model),
    }
    if any(len(values) != len(pairs) for values in metric_values.values()):
        raise ValueError("metric function returned the wrong number of values")

    rows: list[dict[str, object]] = []
    for index, cell in enumerate(pairs):
        values = {metric: metric_values[metric][index] for metric in metric_values}
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(
                f"non-finite metric for {cell.candidate.candidate_id}, cell={cell.cell_idx}"
            )
        rows.append(
            {
                "ratio": cell.candidate.ratio,
                "candidate_pool": cell.candidate.candidate_pool,
                "candidate_id": cell.candidate.candidate_id,
                "bitstring": cell.candidate.bitstring,
                "n_cached": cell.candidate.bitstring.count("1"),
                "hamming_to_primary": cell.candidate.hamming_to_primary,
                "cell_idx": cell.cell_idx,
                "prompt_idx": cell.prompt_idx,
                "prompt": cell.prompt,
                "base_seed": cell.base_seed,
                "seed": cell.seed,
                "candidate_image": str(cell.candidate_image),
                "original_image": str(cell.original_image),
                **values,
            }
        )
    return rows


def write_metric_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty metric table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def merge_metric_tables(paths: Sequence[Path], output: Path) -> list[dict[str, str]]:
    if not paths:
        raise ValueError("at least one metric shard is required")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        fields, shard_rows = _read_table(path, delimiter="\t")
        if fields != list(OUTPUT_FIELDS):
            raise ValueError(f"{path}: unexpected metric columns")
        for row in shard_rows:
            key = (row["candidate_id"], int(row["cell_idx"]))
            if key in seen:
                raise ValueError(f"duplicate metric row {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(
        key=lambda row: (float(row["ratio"]), row["candidate_id"], int(row["cell_idx"]))
    )
    write_metric_rows(output, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cells", type=Path, action="append", required=True)
    evaluate.add_argument("--candidate-tsv", type=Path, action="append", required=True)
    evaluate.add_argument("--candidate-root", type=Path, required=True)
    evaluate.add_argument("--original-cells", type=Path, required=True)
    evaluate.add_argument("--original-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex"
    )
    evaluate.add_argument("--image-reward-model", default="ImageReward-v1.0")
    evaluate.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    merge = commands.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        rows = merge_metric_tables(args.input, args.output)
        print(f"[OK] merged {len(rows)} metric cells -> {args.output}")
        return 0
    pairs = pair_cells(
        cells_paths=args.cells,
        candidate_tsvs=args.candidate_tsv,
        candidate_root=args.candidate_root,
        original_cells_path=args.original_cells,
        original_root=args.original_root,
    )
    device = args.device
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("[WARN] CUDA unavailable; evaluating on CPU", file=sys.stderr)
            device = "cpu"
    rows = evaluate_pairs(
        pairs,
        device=device,
        lpips_net=args.lpips_net,
        image_reward_model=args.image_reward_model,
        clip_model=args.clip_model,
    )
    write_metric_rows(args.output, rows)
    print(f"[OK] evaluated {len(rows)} paired cells -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
