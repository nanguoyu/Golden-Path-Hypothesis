#!/usr/bin/env python3
"""Evaluate paired FLUX Golden Path P6 cells with PSNR, SSIM, and LPIPS."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUT_FIELDS = (
    "candidate_id",
    "candidate_pool",
    "bitstring",
    "n_cached",
    "cell_idx",
    "prompt_idx",
    "seed",
    "candidate_image",
    "original_image",
    "psnr",
    "ssim",
    "lpips",
)

PAIR_PROTOCOL_FIELDS = (
    "prompt",
    "model_id",
    "model_name",
    "dtype",
    "num_steps",
    "width",
    "height",
    "guidance",
)


@dataclass(frozen=True)
class CandidateInfo:
    candidate_id: str
    candidate_pool: str
    bitstring: str


@dataclass(frozen=True)
class PairedCell:
    candidate: CandidateInfo
    cell_idx: int
    prompt_idx: int
    seed: int
    candidate_image: Path
    original_image: Path


def _compute_psnr_ssim(entries):
    from evaluation.eval_metrics import compute_psnr_ssim

    return compute_psnr_ssim(entries, want_psnr=True, want_ssim=True)


def _compute_lpips(entries, device: str, net: str) -> list[float]:
    from evaluation.eval_metrics import compute_lpips

    return compute_lpips(entries, device, net)


def _read_table(path: Path, *, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"table has no header: {path}")
        return list(reader.fieldnames), list(reader)


def load_candidate_info(paths: Sequence[Path]) -> dict[str, CandidateInfo]:
    """Read candidate identity, pool, and bitstring from ordinary TSV shards."""

    if not paths:
        raise ValueError("at least one candidate TSV is required")
    candidates: dict[str, CandidateInfo] = {}
    required = {"candidate_id", "candidate_pool", "bitstring"}
    for path in paths:
        fields, rows = _read_table(path, delimiter="\t")
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{path}: missing candidate columns {sorted(missing)}")
        for row in rows:
            candidate_id = row["candidate_id"].strip()
            pool = row["candidate_pool"].strip()
            bits = row["bitstring"].strip()
            if not candidate_id or not pool:
                raise ValueError(f"{path}: candidate_id and candidate_pool must be non-empty")
            if not bits or set(bits) - {"0", "1"}:
                raise ValueError(f"{path}: invalid bitstring for {candidate_id!r}")
            info = CandidateInfo(candidate_id, pool, bits)
            previous = candidates.get(candidate_id)
            if previous is not None and previous != info:
                raise ValueError(f"conflicting candidate metadata for {candidate_id!r}")
            if previous is not None:
                raise ValueError(f"duplicate candidate_id {candidate_id!r}")
            candidates[candidate_id] = info
    if not candidates:
        raise ValueError("candidate TSVs contain no candidates")
    return candidates


def _cell_key(row: Mapping[str, str], *, source: Path) -> tuple[int, int]:
    try:
        return int(row["prompt_idx"]), int(row["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid prompt_idx/seed row: {dict(row)}") from exc


def _resolve_image(root: Path, row: Mapping[str, str], *, source: Path) -> Path:
    raw = str(row.get("image_path", "")).strip()
    if not raw:
        raise ValueError(f"{source}: cells row has no image_path")
    path = Path(raw)
    return path if path.is_absolute() else Path(root) / path


def _pair_signature(row: Mapping[str, str], *, source: Path) -> tuple[object, ...]:
    try:
        return (
            row["prompt"],
            row["model_id"],
            row["model_name"],
            row["dtype"],
            int(row["num_steps"]),
            int(row["width"]),
            int(row["height"]),
            float(row["guidance"]),
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
    """Join accelerated and original cells by the exact prompt-index/seed key."""

    if len(cells_paths) != len(candidate_tsvs):
        raise ValueError("--cells and --candidate-tsv counts must match")
    candidates = load_candidate_info(candidate_tsvs)

    original_fields, original_rows = _read_table(original_cells_path, delimiter=",")
    required_cells = {
        "candidate_id",
        "cell_idx",
        "prompt_idx",
        "seed",
        "image_path",
        *PAIR_PROTOCOL_FIELDS,
    }
    missing = required_cells.difference(original_fields)
    if missing:
        raise ValueError(f"{original_cells_path}: missing cells columns {sorted(missing)}")

    originals: dict[tuple[int, int], tuple[Path, tuple[object, ...]]] = {}
    for row in original_rows:
        key = _cell_key(row, source=original_cells_path)
        if key in originals:
            raise ValueError(f"{original_cells_path}: duplicate original key {key}")
        try:
            int(row["cell_idx"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{original_cells_path}: invalid cell_idx") from exc
        image = _resolve_image(original_root, row, source=original_cells_path)
        if not image.is_file():
            raise FileNotFoundError(f"missing original image for key {key}: {image}")
        originals[key] = (image, _pair_signature(row, source=original_cells_path))
    if not originals:
        raise ValueError(f"original cells table is empty: {original_cells_path}")

    paired: list[PairedCell] = []
    seen: set[tuple[str, int, int]] = set()
    seen_candidates: set[str] = set()
    for cells_path, candidate_tsv in zip(cells_paths, candidate_tsvs):
        shard_ids = set(load_candidate_info([candidate_tsv]))
        fields, rows = _read_table(cells_path, delimiter=",")
        missing = required_cells.difference(fields)
        if missing:
            raise ValueError(f"{cells_path}: missing cells columns {sorted(missing)}")
        for row in rows:
            candidate_id = row["candidate_id"].strip()
            if candidate_id not in shard_ids:
                raise ValueError(f"{cells_path}: candidate {candidate_id!r} is not in {candidate_tsv}")
            key = _cell_key(row, source=cells_path)
            unique_key = (candidate_id, *key)
            if unique_key in seen:
                raise ValueError(f"duplicate candidate/prompt/seed cell {unique_key}")
            if key not in originals:
                raise ValueError(f"{cells_path}: no original cell for prompt_idx/seed {key}")
            try:
                cell_idx = int(row["cell_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{cells_path}: invalid cell_idx") from exc
            original_image, original_signature = originals[key]
            candidate_signature = _pair_signature(row, source=cells_path)
            if candidate_signature != original_signature:
                raise ValueError(
                    f"{cells_path}: prompt/protocol mismatch for prompt_idx/seed {key}"
                )
            image = _resolve_image(candidate_root, row, source=cells_path)
            if not image.is_file():
                raise FileNotFoundError(f"missing candidate image for {unique_key}: {image}")
            paired.append(
                PairedCell(
                    candidate=candidates[candidate_id],
                    cell_idx=cell_idx,
                    prompt_idx=key[0],
                    seed=key[1],
                    candidate_image=image,
                    original_image=original_image,
                )
            )
            seen.add(unique_key)
            seen_candidates.add(candidate_id)

    missing_candidates = set(candidates).difference(seen_candidates)
    if missing_candidates:
        raise ValueError(f"no cells found for candidates {sorted(missing_candidates)}")
    return sorted(paired, key=lambda cell: (cell.candidate.candidate_id, cell.prompt_idx, cell.seed))


def evaluate_pairs(
    pairs: Sequence[PairedCell],
    *,
    device: str,
    lpips_net: str,
) -> list[dict[str, object]]:
    if not pairs:
        raise ValueError("no paired cells to evaluate")
    entries = [
        (index, cell.candidate_image, cell.original_image, "")
        for index, cell in enumerate(pairs)
    ]
    classic = _compute_psnr_ssim(entries)
    lpips_values = _compute_lpips(entries, device, lpips_net)
    if not (len(pairs) == len(classic["psnr"]) == len(classic["ssim"]) == len(lpips_values)):
        raise ValueError("metric function returned the wrong number of values")

    rows: list[dict[str, object]] = []
    for index, cell in enumerate(pairs):
        values = (classic["psnr"][index], classic["ssim"][index], lpips_values[index])
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(
                f"non-finite metric for {cell.candidate.candidate_id}, "
                f"prompt_idx={cell.prompt_idx}, seed={cell.seed}"
            )
        rows.append(
            {
                "candidate_id": cell.candidate.candidate_id,
                "candidate_pool": cell.candidate.candidate_pool,
                "bitstring": cell.candidate.bitstring,
                "n_cached": cell.candidate.bitstring.count("1"),
                "cell_idx": cell.cell_idx,
                "prompt_idx": cell.prompt_idx,
                "seed": cell.seed,
                "candidate_image": str(cell.candidate_image),
                "original_image": str(cell.original_image),
                "psnr": values[0],
                "ssim": values[1],
                "lpips": values[2],
            }
        )
    return rows


def write_metric_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty metric table")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def merge_metric_tables(paths: Sequence[Path], output: Path) -> list[dict[str, str]]:
    if not paths:
        raise ValueError("at least one metric shard is required")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        fields, shard_rows = _read_table(path, delimiter="\t")
        if fields != list(OUTPUT_FIELDS):
            raise ValueError(f"{path}: unexpected metric columns")
        for row in shard_rows:
            key = (row["candidate_id"], int(row["prompt_idx"]), int(row["seed"]))
            if key in seen:
                raise ValueError(f"duplicate metric row {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["candidate_id"], int(row["prompt_idx"]), int(row["seed"])))
    write_metric_rows(output, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("evaluate", help="evaluate one or more generation shards")
    evaluate.add_argument("--cells", type=Path, action="append", required=True)
    evaluate.add_argument("--candidate-tsv", type=Path, action="append", required=True)
    evaluate.add_argument("--candidate-root", type=Path, required=True)
    evaluate.add_argument("--original-cells", type=Path, required=True)
    evaluate.add_argument("--original-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")

    merge = commands.add_parser("merge", help="merge disjoint per-worker metric TSVs")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        rows = merge_metric_tables(args.input, args.output)
        print(f"merged {len(rows)} metric cells -> {args.output}")
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
            print("[WARN] CUDA unavailable; evaluating LPIPS on CPU", file=sys.stderr)
            device = "cpu"
    rows = evaluate_pairs(pairs, device=device, lpips_net=args.lpips_net)
    write_metric_rows(args.output, rows)
    print(f"evaluated {len(rows)} paired cells -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
