#!/usr/bin/env python3
"""Strict same-seed Stage C2 evaluator for completed HunyuanVideo tasks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hunyuan_video import config as hv_config  # noqa: E402
from hunyuan_video.config import FormalRow, FormalTask, TaskManifest, hash_json  # noqa: E402
from hunyuan_video.records import (  # noqa: E402
    load_self_hashed_json,
    validate_attempt_record,
    write_immutable_json,
)


REQUIRED_ROLES = ("video", "action", "generation_metadata")
DEFAULT_LPIPS_BATCH_SIZE = 16
VBENCH_METADATA_PATH = ROOT / "resources/hunyuan_video/vbench_metadata.v1.json"
PairKey = tuple[str, int, int]
ReaderFactory = Callable[[Path], Any]
Metric = Callable[..., float]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict HunyuanVideo Stage C2 evaluator")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--candidate-row-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips-batch-size", type=_positive_int, default=DEFAULT_LPIPS_BATCH_SIZE)
    return parser.parse_args(argv)


def _load_self_hashed_manifest(path: Path, *, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported local manifest schema: {payload.get('schema')}")
    expected = payload.get("manifest_sha256")
    if hash_json({key: value for key, value in payload.items() if key != "manifest_sha256"}) != expected:
        raise ValueError(f"local manifest self-hash mismatch: {path}")
    return payload


def _manifest_value(section: Mapping[str, Any], key: str) -> Any:
    field = section.get(key)
    if not isinstance(field, Mapping) or "value" not in field:
        raise ValueError(f"protocol field lacks a frozen value: {key}")
    return field["value"]


def _load_local_bindings(
    manifest: TaskManifest,
    candidate_row: FormalRow,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate_row.stage in {"C4", "E7"} and candidate_row.split == "VBench":
        vbench = _load_self_hashed_manifest(
            VBENCH_METADATA_PATH,
            schema="hunyuan_video.vbench_metadata.v1",
        )
        artifacts = vbench.get("artifacts", [])
        prompt_ids = [row["artifact_id"] for row in artifacts]
        dataset = {
            "manifest_sha256": vbench["manifest_sha256"],
            "prompts": [
                {"prompt_id": row["artifact_id"], "prompt": row["prompt"]}
                for row in artifacts
            ],
            "splits": {
                "VBench": {
                    "count": len(prompt_ids),
                    "prompt_ids": prompt_ids,
                    "prompt_ids_sha256": hash_json(prompt_ids),
                }
            },
        }
    elif candidate_row.stage in {"E7", "EE"} and candidate_row.split == "PENGUIN599":
        dataset = _load_self_hashed_manifest(
            hv_config.DATASET_PATH,
            schema="hunyuan_video.dataset_splits.v2",
        )
        prompts = dataset.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != 599:
            raise ValueError(
                "E7/EE Penguin evaluation requires the complete 599-prompt table"
            )
        prompt_ids = [row.get("prompt_id") for row in prompts if isinstance(row, Mapping)]
        if (
            len(prompt_ids) != len(prompts)
            or any(not isinstance(prompt_id, str) or not prompt_id for prompt_id in prompt_ids)
            or len(set(prompt_ids)) != len(prompt_ids)
        ):
            raise ValueError("E7/EE Penguin prompt identity mismatch")
        dataset = dict(dataset)
        dataset["splits"] = {
            "PENGUIN599": {
                "count": len(prompt_ids),
                "prompt_ids": prompt_ids,
                "prompt_ids_sha256": hash_json(prompt_ids),
            }
        }
    else:
        dataset = _load_self_hashed_manifest(
            hv_config.DATASET_PATH,
            schema="hunyuan_video.dataset_splits.v2",
        )
    if dataset["manifest_sha256"] != manifest.dataset_manifest_sha256:
        raise ValueError("task manifest dataset binding differs from the local dataset manifest")

    protocols, protocol_row = hv_config.load_protocol_registry(candidate_row.protocol_id)
    expected_protocol_manifest = manifest.protocol_manifest_sha256s.get(candidate_row.protocol_id)
    if protocols["manifest_sha256"] != expected_protocol_manifest:
        raise ValueError("task manifest protocol binding differs from the local protocol manifest")
    return dataset, protocol_row


def _row_by_id(manifest: TaskManifest, row_id: str) -> FormalRow:
    matches = [row for row in manifest.formal_rows if row.row_id == row_id]
    if len(matches) != 1:
        raise ValueError(f"task manifest must contain exactly one formal row {row_id!r}")
    return matches[0]


def _tasks_by_key(tasks: list[FormalTask], *, row_id: str) -> dict[PairKey, FormalTask]:
    indexed: dict[PairKey, FormalTask] = {}
    for task in tasks:
        run = task.run_spec
        key = (run.prompt_id, run.seed, run.repeat)
        if key in indexed:
            raise ValueError(f"duplicate (prompt_id, seed, repeat) in row {row_id}: {key}")
        indexed[key] = task
    return indexed


def _pair_tasks(
    manifest: TaskManifest,
    candidate_row_id: str,
    dataset: Mapping[str, Any],
    *,
    expected_stage: str,
) -> tuple[FormalRow, FormalRow, list[tuple[PairKey, FormalTask, FormalTask]]]:
    candidate_row = _row_by_id(manifest, candidate_row_id)
    if candidate_row.stage != expected_stage:
        raise ValueError(f"candidate FormalRow phase must be {expected_stage}")
    if not candidate_row.reference_row_id or candidate_row.reference_row_id == candidate_row.row_id:
        raise ValueError("candidate FormalRow must declare a distinct reference_row_id")
    reference_row = _row_by_id(manifest, candidate_row.reference_row_id)
    for field in ("stage", "protocol_id", "source_protocol_id", "split"):
        if getattr(candidate_row, field) != getattr(reference_row, field):
            raise ValueError(f"candidate/reference {field} identity mismatch")

    split = dataset.get("splits", {}).get(candidate_row.split)
    if not isinstance(split, Mapping):
        raise ValueError(f"local dataset lacks split {candidate_row.split!r}")
    prompt_ids = split.get("prompt_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("dataset split prompt IDs must be a non-empty unique list")
    if split.get("count") != len(prompt_ids) or split.get("prompt_ids_sha256") != hash_json(prompt_ids):
        raise ValueError("dataset split identity mismatch")

    candidate_tasks = [task for task in manifest.tasks if task.formal_row_id == candidate_row.row_id]
    reference_tasks = [task for task in manifest.tasks if task.formal_row_id == reference_row.row_id]
    candidates = _tasks_by_key(candidate_tasks, row_id=candidate_row.row_id)
    references = _tasks_by_key(reference_tasks, row_id=reference_row.row_id)
    expected = {
        (str(prompt_id), seed, repeat)
        for prompt_id in prompt_ids
        for seed in candidate_row.seeds
        for repeat in range(candidate_row.repeats)
    }
    reference_expected = {
        (str(prompt_id), seed, repeat)
        for prompt_id in prompt_ids
        for seed in reference_row.seeds
        for repeat in range(reference_row.repeats)
    }
    if set(candidates) != expected:
        raise ValueError("candidate row key set differs from its frozen dataset split/seeds/repeats")
    if set(references) != reference_expected:
        raise ValueError("reference row key set differs from its frozen dataset split/seeds/repeats")
    if set(candidates) != set(references):
        raise ValueError("candidate/reference key sets differ")

    prompt_rows = dataset.get("prompts")
    if not isinstance(prompt_rows, list):
        raise ValueError("dataset prompts must be a list")
    prompt_by_id: dict[str, str] = {}
    for row in prompt_rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id or prompt_id in prompt_by_id:
            raise ValueError("dataset prompt IDs must be unique and non-empty")
        prompt_by_id[prompt_id] = str(row.get("prompt", ""))

    pairs: list[tuple[PairKey, FormalTask, FormalTask]] = []
    for key in sorted(expected):
        candidate = candidates[key]
        reference = references[key]
        local_prompt = prompt_by_id.get(key[0])
        if local_prompt is None or candidate.run_spec.prompt != local_prompt:
            raise ValueError(f"candidate prompt differs from local dataset for {key[0]}")
        if reference.run_spec.prompt != local_prompt:
            raise ValueError(f"candidate/reference prompt identity mismatch for {key[0]}")
        if candidate.run_spec.phase != reference.run_spec.phase:
            raise ValueError(f"candidate/reference phase identity mismatch for {key}")
        if candidate.run_spec.protocol_id != reference.run_spec.protocol_id:
            raise ValueError(f"candidate/reference protocol identity mismatch for {key}")
        pairs.append((key, candidate, reference))
    return candidate_row, reference_row, pairs


def _artifact_identity(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": row["role"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in completion["artifacts"]
    ]


def _require_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{label} {field} binding mismatch")


def _resolve_accepted_attempt(
    output_root: Path,
    manifest: TaskManifest,
    row: FormalRow,
    task: FormalTask,
    protocol_sha256: str,
) -> dict[str, Any]:
    run = task.run_spec
    task_root = output_root / run.phase / row.row_id / run.task_id
    marker_path = task_root / "accepted_attempt.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"frozen accepted-attempt marker is missing: {marker_path}")
    marker = load_self_hashed_json(marker_path, "accepted_attempt_sha256")
    _require_fields(
        marker,
        {
            "schema": "hunyuan_video.stage_c_accepted_attempt.v1",
            "task_id": run.task_id,
            "task_manifest_sha256": manifest.sha256,
            "run_spec_sha256": run.sha256,
        },
        label="accepted-attempt",
    )
    assignment_sha256 = str(marker.get("assignment_manifest_sha256", ""))
    attempt_id = marker.get("accepted_attempt_id")
    if isinstance(attempt_id, bool) or attempt_id not in (0, 1):
        raise ValueError("accepted-attempt ID must be 0 or 1")
    attempt_dir = task_root / f"attempt_{attempt_id:03d}"
    completion = validate_attempt_record(
        attempt_dir / "generation_done.json",
        task_id=run.task_id,
        task_manifest_sha256=manifest.sha256,
        assignment_manifest_sha256=assignment_sha256,
        run_spec_sha256=run.sha256,
        required_roles=REQUIRED_ROLES,
    )
    if completion["manifest_sha256"] != marker.get("generation_done_sha256"):
        raise ValueError("accepted-attempt generation_done binding mismatch")
    if hash_json(_artifact_identity(completion)) != marker.get("artifact_identity_sha256"):
        raise ValueError("accepted-attempt artifact identity mismatch")

    artifacts = {row["role"]: row for row in completion["artifacts"]}
    action = load_self_hashed_json(attempt_dir / artifacts["action"]["relative_path"], "manifest_sha256")
    _require_fields(
        action,
        {
            "schema": "hunyuan_video.stage_c_actions.v1",
            "task_id": run.task_id,
            "run_spec_sha256": run.sha256,
            "formal_row_id": row.row_id,
            "protocol_sha256": protocol_sha256,
        },
        label="action",
    )
    metadata = load_self_hashed_json(
        attempt_dir / artifacts["generation_metadata"]["relative_path"],
        "manifest_sha256",
    )
    _require_fields(
        metadata,
        {
            "schema": "hunyuan_video.stage_c_generation_metadata.v1",
            "task_id": run.task_id,
            "run_spec_sha256": run.sha256,
            "formal_row_id": row.row_id,
            "formal_row_sha256": row.sha256,
            "task_manifest_sha256": manifest.sha256,
            "assignment_manifest_sha256": assignment_sha256,
            "attempt_index": attempt_id,
            "protocol_manifest_sha256": manifest.protocol_manifest_sha256s[row.protocol_id],
            "protocol_sha256": protocol_sha256,
        },
        label="generation-metadata",
    )
    video_metadata = metadata.get("video")
    video_artifact = artifacts["video"]
    if not isinstance(video_metadata, Mapping):
        raise ValueError("generation-metadata video binding is missing")
    _require_fields(
        video_metadata,
        {
            "path": video_artifact["relative_path"],
            "bytes": video_artifact["bytes"],
            "sha256": video_artifact["sha256"],
        },
        label="generation-metadata video",
    )
    has_latent = "final_latent" in artifacts
    if "final_latent" in metadata and (metadata["final_latent"] is not None) != has_latent:
        raise ValueError("generation-metadata final_latent binding mismatch")
    return {
        "marker": marker,
        "completion": completion,
        "artifacts": artifacts,
        "attempt_dir": attempt_dir,
    }


def _open_video_reader(path: Path) -> Any:
    import imageio.v2 as imageio

    return imageio.get_reader(str(path), format="ffmpeg")


def _load_lpips_model(device: str) -> Any:
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def _default_metrics() -> tuple[Metric, Metric]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    return peak_signal_noise_ratio, structural_similarity


def _fps(reader: Any, path: Path, expected: int) -> None:
    metadata = reader.get_meta_data()
    observed = metadata.get("fps") if isinstance(metadata, Mapping) else None
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError(f"video metadata has invalid fps: {path}")
    if not math.isfinite(float(observed)) or float(observed) != float(expected):
        raise ValueError(f"video fps mismatch for {path}: {observed} != {expected}")


def _frame(frame: Any, *, path: Path, index: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(frame)
    if array.shape != (height, width, 3):
        raise ValueError(f"video frame shape mismatch for {path} at {index}: {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"video frame dtype mismatch for {path} at {index}: {array.dtype}")
    return array


def _lpips_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).div(127.5).sub(1.0)


def _finite(value: Any, label: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric value: {label}")
    return result


def _psnr_value(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    metric: Metric,
    index: int,
) -> float:
    value = metric(reference, candidate, data_range=255)
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if math.isfinite(result):
        return result
    if result == math.inf and np.array_equal(reference, candidate):
        # One uint8 least-significant-unit error is the smallest observable
        # nonzero frame MSE. Use its PSNR as the finite exact-match ceiling.
        return 10.0 * math.log10((255.0**2) * reference.size)
    raise ValueError(f"non-finite metric value: psnr[{index}]")


def _lpips_batch_values(value: Any, expected: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        if flat.numel() != expected:
            raise ValueError(f"LPIPS batch output count mismatch: {flat.numel()} != {expected}")
        raw_values = flat.cpu().tolist()
    else:
        flat = np.asarray(value).reshape(-1)
        if flat.size != expected:
            raise ValueError(f"LPIPS batch output count mismatch: {flat.size} != {expected}")
        raw_values = flat.tolist()
    return [_finite(item, f"lpips_batch[{index}]") for index, item in enumerate(raw_values)]


def _validate_lpips_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lpips_batch_size must be a positive integer")
    return value


def _stream_metrics(
    candidate_path: Path,
    reference_path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    fps: int,
    device: str,
    reader_factory: ReaderFactory,
    lpips_model: Any,
    psnr_metric: Metric,
    ssim_metric: Metric,
    lpips_batch_size: int,
) -> dict[str, list[float]]:
    lpips_batch_size = _validate_lpips_batch_size(lpips_batch_size)
    try:
        candidate_reader = reader_factory(candidate_path)
    except Exception as error:
        raise ValueError("imageio could not open a paired video") from error
    try:
        reference_reader = reader_factory(reference_path)
    except Exception as error:
        candidate_reader.close()
        raise ValueError("imageio could not open a paired video") from error
    sentinel = object()
    lpips_values: list[float | None] = [None] * frames
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    candidate_temporal: list[float | None] = [None] * (frames - 1)
    reference_temporal: list[float | None] = [None] * (frames - 1)
    pending_left: list[torch.Tensor] = []
    pending_right: list[torch.Tensor] = []
    pending_destinations: list[tuple[str, int]] = []
    previous_candidate: torch.Tensor | None = None
    previous_reference: torch.Tensor | None = None

    def flush_lpips() -> None:
        if not pending_left:
            return
        values = _lpips_batch_values(
            lpips_model(torch.cat(pending_left, dim=0), torch.cat(pending_right, dim=0)),
            len(pending_destinations),
        )
        for (kind, index), value in zip(pending_destinations, values):
            if kind == "pair":
                lpips_values[index] = value
            elif kind == "candidate_temporal":
                candidate_temporal[index] = value
            else:
                reference_temporal[index] = value
        pending_left.clear()
        pending_right.clear()
        pending_destinations.clear()

    def enqueue_lpips(left: torch.Tensor, right: torch.Tensor, kind: str, index: int) -> None:
        pending_left.append(left)
        pending_right.append(right)
        pending_destinations.append((kind, index))
        if len(pending_destinations) == lpips_batch_size:
            flush_lpips()

    try:
        _fps(candidate_reader, candidate_path, fps)
        _fps(reference_reader, reference_path, fps)
        with torch.no_grad():
            for index, (candidate_raw, reference_raw) in enumerate(
                zip_longest(candidate_reader, reference_reader, fillvalue=sentinel)
            ):
                if candidate_raw is sentinel or reference_raw is sentinel:
                    raise ValueError("candidate/reference video frame-count mismatch")
                if index >= frames:
                    raise ValueError(f"video frame-count exceeds expected {frames}")
                candidate = _frame(
                    candidate_raw,
                    path=candidate_path,
                    index=index,
                    height=height,
                    width=width,
                )
                reference = _frame(
                    reference_raw,
                    path=reference_path,
                    index=index,
                    height=height,
                    width=width,
                )
                candidate_tensor = _lpips_tensor(candidate, device)
                reference_tensor = _lpips_tensor(reference, device)
                enqueue_lpips(candidate_tensor, reference_tensor, "pair", index)
                psnr_values.append(
                    _psnr_value(
                        reference,
                        candidate,
                        metric=psnr_metric,
                        index=index,
                    )
                )
                ssim_values.append(
                    _finite(
                        ssim_metric(reference, candidate, data_range=255, channel_axis=-1),
                        f"ssim[{index}]",
                    )
                )
                if previous_candidate is not None and previous_reference is not None:
                    enqueue_lpips(
                        previous_candidate,
                        candidate_tensor,
                        "candidate_temporal",
                        index - 1,
                    )
                    enqueue_lpips(
                        previous_reference,
                        reference_tensor,
                        "reference_temporal",
                        index - 1,
                    )
                previous_candidate = candidate_tensor
                previous_reference = reference_tensor
            flush_lpips()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("imageio failed while decoding a paired video") from error
    finally:
        candidate_reader.close()
        reference_reader.close()
    if len(psnr_values) != frames:
        raise ValueError(f"video frame-count mismatch: {len(psnr_values)} != {frames}")
    if any(value is None for value in lpips_values):
        raise ValueError("LPIPS metric count differs from protocol frame count")
    if any(value is None for value in candidate_temporal + reference_temporal):
        raise ValueError("temporal metric count differs from protocol frame count")
    ordered_lpips = [float(value) for value in lpips_values if value is not None]
    temporal_values = [
        abs(float(candidate) - float(reference))
        for candidate, reference in zip(candidate_temporal, reference_temporal)
        if candidate is not None and reference is not None
    ]
    return {
        "lpips": ordered_lpips,
        "psnr": psnr_values,
        "ssim": ssim_values,
        "temporal_lpips_delta": temporal_values,
    }


def _load_latent(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"final latent is not a tensor: {path}")
    return value


def _latent_rms(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    latent_loader: Callable[[Path], torch.Tensor],
) -> float | None:
    candidate_row = candidate["artifacts"].get("final_latent")
    reference_row = reference["artifacts"].get("final_latent")
    if candidate_row is None and reference_row is None:
        return None
    if candidate_row is None or reference_row is None:
        raise ValueError("final_latent must be present for both paired tasks or neither")
    candidate_latent = latent_loader(candidate["attempt_dir"] / candidate_row["relative_path"])
    reference_latent = latent_loader(reference["attempt_dir"] / reference_row["relative_path"])
    if candidate_latent.shape != reference_latent.shape or candidate_latent.dtype != reference_latent.dtype:
        raise ValueError("paired final_latent shape/dtype mismatch")
    if candidate_latent.numel() == 0:
        raise ValueError("paired final_latent tensors must be non-empty")
    candidate_float = candidate_latent.detach().to(dtype=torch.float64, device="cpu")
    reference_float = reference_latent.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(candidate_float).all() or not torch.isfinite(reference_float).all():
        raise ValueError("paired final_latent contains non-finite values")
    return _finite(torch.sqrt(torch.mean(torch.square(candidate_float - reference_float))), "latent_rms")


def _mean(values: list[float], label: str) -> float:
    if not values:
        raise ValueError(f"cannot compute empty mean: {label}")
    return _finite(math.fsum(values) / len(values), f"{label}_mean")


def _pair_shard_bounds(total: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValueError("shard_index must be an integer")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if shard_count > total:
        raise ValueError("shard_count cannot exceed the formal pair count")
    return total * shard_index // shard_count, total * (shard_index + 1) // shard_count


def _prepare_formal_evaluation(
    *,
    task_manifest_path: Path,
    candidate_row_id: str,
    expected_stage: str,
) -> dict[str, Any]:
    manifest = TaskManifest.load(task_manifest_path)
    initial_candidate_row = _row_by_id(manifest, candidate_row_id)
    dataset, protocol_row = _load_local_bindings(manifest, initial_candidate_row)
    candidate_row, reference_row, task_pairs = _pair_tasks(
        manifest,
        candidate_row_id,
        dataset,
        expected_stage=expected_stage,
    )

    generation = protocol_row.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("local protocol generation section is missing")
    height = int(_manifest_value(generation, "height"))
    width = int(_manifest_value(generation, "width"))
    frames = int(_manifest_value(generation, "frames"))
    fps = int(_manifest_value(generation, "fps"))
    if min(height, width, frames, fps) <= 0:
        raise ValueError("protocol geometry, frame count, and fps must be positive")
    protocol_sha256 = hash_json(protocol_row)
    identity = {
        "schema": f"hunyuan_video.stage_{expected_stage.lower()}_evaluation.v1",
        "task_manifest_sha256": manifest.sha256,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "protocol_manifest_sha256": manifest.protocol_manifest_sha256s[
            candidate_row.protocol_id
        ],
        "protocol_sha256": protocol_sha256,
        "candidate_row_id": candidate_row.row_id,
        "candidate_row_sha256": candidate_row.sha256,
        "reference_row_id": reference_row.row_id,
        "reference_row_sha256": reference_row.sha256,
        "protocol_id": candidate_row.protocol_id,
        "split": candidate_row.split,
        "phase": candidate_row.stage,
    }
    return {
        "manifest": manifest,
        "candidate_row": candidate_row,
        "reference_row": reference_row,
        "task_pairs": task_pairs,
        "height": height,
        "width": width,
        "frames": frames,
        "fps": fps,
        "protocol_sha256": protocol_sha256,
        "identity": identity,
    }


def _aggregate_pair_means(
    pair_records: list[dict[str, Any]],
) -> dict[str, float | None]:
    aggregate_values: dict[str, list[float]] = {
        "lpips": [],
        "psnr": [],
        "ssim": [],
        "temporal_lpips_delta": [],
    }
    latent_values: list[float] = []
    for pair in pair_records:
        expected_pair_sha256 = pair.get("pair_sha256")
        pair_body = {key: value for key, value in pair.items() if key != "pair_sha256"}
        if hash_json(pair_body) != expected_pair_sha256:
            raise ValueError("paired metric record self-hash mismatch")
        metrics = pair.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("paired metric record lacks metrics")
        for name in aggregate_values:
            values = metrics.get(name)
            if not isinstance(values, list) or not values:
                raise ValueError(f"paired metric record lacks {name}")
            aggregate_values[name].extend(_finite(value, name) for value in values)
        latent_rms = metrics.get("latent_rms")
        if latent_rms is not None:
            latent_values.append(_finite(latent_rms, "latent_rms"))
    return {
        "lpips": _mean(aggregate_values["lpips"], "aggregate_lpips"),
        "psnr": _mean(aggregate_values["psnr"], "aggregate_psnr"),
        "ssim": _mean(aggregate_values["ssim"], "aggregate_ssim"),
        "temporal_lpips_delta": _mean(
            aggregate_values["temporal_lpips_delta"], "aggregate_temporal_lpips_delta"
        ),
        "latent_rms": _mean(latent_values, "aggregate_latent_rms") if latent_values else None,
    }


def _evaluation_result(
    *,
    identity: Mapping[str, Any],
    pair_records: list[dict[str, Any]],
    lpips_batch_size: int,
) -> dict[str, Any]:
    result = {
        **identity,
        "lpips_batch_size": lpips_batch_size,
        "pair_count": len(pair_records),
        "pairs": pair_records,
        "means": _aggregate_pair_means(pair_records),
    }
    result["manifest_sha256"] = hash_json(result)
    return result


def _evaluate_formal_pairs_impl(
    *,
    output_root: Path,
    task_manifest_path: Path,
    candidate_row_id: str,
    output: Path,
    device: str,
    reader_factory: ReaderFactory | None = None,
    lpips_model: Any | None = None,
    psnr_metric: Metric | None = None,
    ssim_metric: Metric | None = None,
    lpips_batch_size: int = DEFAULT_LPIPS_BATCH_SIZE,
    latent_loader: Callable[[Path], torch.Tensor] = _load_latent,
    expected_stage: str = "C2",
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    lpips_batch_size = _validate_lpips_batch_size(lpips_batch_size)
    prepared = _prepare_formal_evaluation(
        task_manifest_path=task_manifest_path,
        candidate_row_id=candidate_row_id,
        expected_stage=expected_stage,
    )
    manifest = prepared["manifest"]
    candidate_row = prepared["candidate_row"]
    reference_row = prepared["reference_row"]
    all_task_pairs = prepared["task_pairs"]
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_index is None:
        pair_start, pair_stop = 0, len(all_task_pairs)
    else:
        assert shard_count is not None
        pair_start, pair_stop = _pair_shard_bounds(
            len(all_task_pairs), shard_index, shard_count
        )
    task_pairs = all_task_pairs[pair_start:pair_stop]

    reader_factory = reader_factory or _open_video_reader
    if lpips_model is None:
        lpips_model = _load_lpips_model(device)
    if psnr_metric is None or ssim_metric is None:
        default_psnr, default_ssim = _default_metrics()
        psnr_metric = psnr_metric or default_psnr
        ssim_metric = ssim_metric or default_ssim

    pair_records: list[dict[str, Any]] = []
    for key, candidate_task, reference_task in task_pairs:
        candidate = _resolve_accepted_attempt(
            output_root,
            manifest,
            candidate_row,
            candidate_task,
            prepared["protocol_sha256"],
        )
        reference = _resolve_accepted_attempt(
            output_root,
            manifest,
            reference_row,
            reference_task,
            prepared["protocol_sha256"],
        )
        candidate_video = candidate["artifacts"]["video"]
        reference_video = reference["artifacts"]["video"]
        metrics = _stream_metrics(
            candidate["attempt_dir"] / candidate_video["relative_path"],
            reference["attempt_dir"] / reference_video["relative_path"],
            frames=prepared["frames"],
            height=prepared["height"],
            width=prepared["width"],
            fps=prepared["fps"],
            device=device,
            reader_factory=reader_factory,
            lpips_model=lpips_model,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            lpips_batch_size=lpips_batch_size,
        )
        latent_rms = _latent_rms(candidate, reference, latent_loader=latent_loader)
        pair_record: dict[str, Any] = {
            "schema": f"hunyuan_video.stage_{expected_stage.lower()}_pair.v1",
            "prompt_id": key[0],
            "seed": key[1],
            "repeat": key[2],
            "candidate": {
                "row_id": candidate_row.row_id,
                "task_id": candidate_task.run_spec.task_id,
                "run_spec_sha256": candidate_task.run_spec.sha256,
                "accepted_attempt_sha256": candidate["marker"]["accepted_attempt_sha256"],
                "generation_done_sha256": candidate["completion"]["manifest_sha256"],
                "video_sha256": candidate_video["sha256"],
                "final_latent_sha256": candidate["artifacts"].get("final_latent", {}).get("sha256"),
            },
            "reference": {
                "row_id": reference_row.row_id,
                "task_id": reference_task.run_spec.task_id,
                "run_spec_sha256": reference_task.run_spec.sha256,
                "accepted_attempt_sha256": reference["marker"]["accepted_attempt_sha256"],
                "generation_done_sha256": reference["completion"]["manifest_sha256"],
                "video_sha256": reference_video["sha256"],
                "final_latent_sha256": reference["artifacts"].get("final_latent", {}).get("sha256"),
            },
            "metrics": {**metrics, "latent_rms": latent_rms},
            "means": {
                "lpips": _mean(metrics["lpips"], "pair_lpips"),
                "psnr": _mean(metrics["psnr"], "pair_psnr"),
                "ssim": _mean(metrics["ssim"], "pair_ssim"),
                "temporal_lpips_delta": _mean(
                    metrics["temporal_lpips_delta"], "pair_temporal_lpips_delta"
                ),
                "latent_rms": latent_rms,
            },
        }
        pair_record["pair_sha256"] = hash_json(pair_record)
        pair_records.append(pair_record)

    if shard_index is None:
        result = _evaluation_result(
            identity=prepared["identity"],
            pair_records=pair_records,
            lpips_batch_size=lpips_batch_size,
        )
    else:
        assert shard_count is not None
        result = {
            **prepared["identity"],
            "schema": f"hunyuan_video.stage_{expected_stage.lower()}_evaluation_shard.v1",
            "lpips_batch_size": lpips_batch_size,
            "total_pair_count": len(all_task_pairs),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "pair_start": pair_start,
            "pair_stop": pair_stop,
            "pair_count": len(pair_records),
            "pairs": pair_records,
        }
        result["manifest_sha256"] = hash_json(result)
    write_immutable_json(output, result)
    return result


def evaluate_formal_pairs(
    *,
    output_root: Path,
    task_manifest_path: Path,
    candidate_row_id: str,
    output: Path,
    device: str,
    reader_factory: ReaderFactory | None = None,
    lpips_model: Any | None = None,
    psnr_metric: Metric | None = None,
    ssim_metric: Metric | None = None,
    lpips_batch_size: int = DEFAULT_LPIPS_BATCH_SIZE,
    latent_loader: Callable[[Path], torch.Tensor] = _load_latent,
    expected_stage: str = "C2",
) -> dict[str, Any]:
    return _evaluate_formal_pairs_impl(
        output_root=output_root,
        task_manifest_path=task_manifest_path,
        candidate_row_id=candidate_row_id,
        output=output,
        device=device,
        reader_factory=reader_factory,
        lpips_model=lpips_model,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric,
        lpips_batch_size=lpips_batch_size,
        latent_loader=latent_loader,
        expected_stage=expected_stage,
    )


def evaluate_formal_pair_shard(
    *,
    output_root: Path,
    task_manifest_path: Path,
    candidate_row_id: str,
    output: Path,
    device: str,
    shard_index: int,
    shard_count: int,
    reader_factory: ReaderFactory | None = None,
    lpips_model: Any | None = None,
    psnr_metric: Metric | None = None,
    ssim_metric: Metric | None = None,
    lpips_batch_size: int = DEFAULT_LPIPS_BATCH_SIZE,
    latent_loader: Callable[[Path], torch.Tensor] = _load_latent,
    expected_stage: str = "C2",
) -> dict[str, Any]:
    return _evaluate_formal_pairs_impl(
        output_root=output_root,
        task_manifest_path=task_manifest_path,
        candidate_row_id=candidate_row_id,
        output=output,
        device=device,
        shard_index=shard_index,
        shard_count=shard_count,
        reader_factory=reader_factory,
        lpips_model=lpips_model,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric,
        lpips_batch_size=lpips_batch_size,
        latent_loader=latent_loader,
        expected_stage=expected_stage,
    )


def merge_formal_pair_shards(
    *,
    task_manifest_path: Path,
    candidate_row_id: str,
    shard_paths: list[Path],
    output: Path,
    expected_stage: str,
) -> dict[str, Any]:
    if not shard_paths:
        raise ValueError("at least one paired-evaluation shard is required")
    prepared = _prepare_formal_evaluation(
        task_manifest_path=task_manifest_path,
        candidate_row_id=candidate_row_id,
        expected_stage=expected_stage,
    )
    expected_pairs = prepared["task_pairs"]
    shard_count = len(shard_paths)
    pair_records: list[dict[str, Any]] = []
    lpips_batch_size: int | None = None
    identity = prepared["identity"]
    for shard_index, path in enumerate(shard_paths):
        shard = load_self_hashed_json(path, "manifest_sha256")
        expected_schema = f"hunyuan_video.stage_{expected_stage.lower()}_evaluation_shard.v1"
        if shard.get("schema") != expected_schema:
            raise ValueError("paired-evaluation shard schema mismatch")
        for key, value in identity.items():
            if key == "schema":
                continue
            if shard.get(key) != value:
                raise ValueError(f"paired-evaluation shard identity mismatch: {key}")
        if (
            shard.get("shard_index") != shard_index
            or shard.get("shard_count") != shard_count
        ):
            raise ValueError("paired-evaluation shard position mismatch")
        pair_start, pair_stop = _pair_shard_bounds(
            len(expected_pairs), shard_index, shard_count
        )
        if (
            shard.get("total_pair_count") != len(expected_pairs)
            or shard.get("pair_start") != pair_start
            or shard.get("pair_stop") != pair_stop
        ):
            raise ValueError("paired-evaluation shard range mismatch")
        records = shard.get("pairs")
        if not isinstance(records, list) or shard.get("pair_count") != len(records):
            raise ValueError("paired-evaluation shard pair count mismatch")
        expected_keys = [key for key, _, _ in expected_pairs[pair_start:pair_stop]]
        observed_keys = [
            (record.get("prompt_id"), record.get("seed"), record.get("repeat"))
            for record in records
            if isinstance(record, Mapping)
        ]
        if observed_keys != expected_keys or len(observed_keys) != len(records):
            raise ValueError("paired-evaluation shard pair order mismatch")
        current_batch_size = shard.get("lpips_batch_size")
        if lpips_batch_size is None:
            lpips_batch_size = current_batch_size
        elif current_batch_size != lpips_batch_size:
            raise ValueError("paired-evaluation shard LPIPS batch-size mismatch")
        pair_records.extend(records)
    if len(pair_records) != len(expected_pairs):
        raise ValueError("merged paired-evaluation coverage mismatch")
    if isinstance(lpips_batch_size, bool) or not isinstance(lpips_batch_size, int):
        raise ValueError("paired-evaluation shard lacks LPIPS batch size")
    result = _evaluation_result(
        identity=identity,
        pair_records=pair_records,
        lpips_batch_size=lpips_batch_size,
    )
    write_immutable_json(output, result)
    return result


def evaluate_stage_c2(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible Stage C2 entrypoint."""

    return evaluate_formal_pairs(expected_stage="C2", **kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_stage_c2(
        output_root=args.output_root,
        task_manifest_path=args.task_manifest,
        candidate_row_id=args.candidate_row_id,
        output=args.output,
        device=args.device,
        lpips_batch_size=args.lpips_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
