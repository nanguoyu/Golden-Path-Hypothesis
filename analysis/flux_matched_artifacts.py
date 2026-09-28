#!/usr/bin/env python3
"""Shared validation for frozen FLUX matched-baseline artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


METRICS = ("psnr", "ssim", "lpips", "clip", "image_reward")
INTERVAL_CACHED_STEPS = {2: 23, 3: 31, 5: 37, 8: 41, 9: 41}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc


def _path(value: Any) -> Path:
    return Path(str(value)).resolve()


def _optional(value: Any) -> str:
    value = str(value)
    return "" if value in ("", "-") else value


def _same_number(actual: Any, expected: Any, name: str) -> None:
    expected = _optional(expected)
    if not expected:
        return
    if actual is None or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(row: Mapping[str, Any]) -> Path:
    return _path(row["output_root"]) / str(row["run_name"])


def image_count(path: Path) -> int:
    return sum(1 for item in path.glob("img_*.png") if item.is_file())


def validate_original(row: Mapping[str, Any]) -> dict[str, Any]:
    root = _path(row["original_dir"])
    expected = int(row["n_prompts"])
    if image_count(root) != expected:
        raise ValueError(f"{root}: original image coverage mismatch")
    timing = _load_json(root / "timing.json")
    if timing.get("mode_raw") != "original":
        raise ValueError(f"{root}: reference is not mode=original")
    if int(timing.get("base_seed", -1)) != 42 or int(timing.get("num_steps", -1)) != 50:
        raise ValueError(f"{root}: original seed/steps mismatch")
    if int(timing.get("n_images", -1)) != expected:
        raise ValueError(f"{root}: original timing image count mismatch")
    return timing


def _validate_payload_decision_sample(
    root: Path, *, expected_count: int, steps: int, locked: bool
) -> None:
    paths = sorted(root.glob("decisions_*.json"))
    if len(paths) != expected_count:
        raise ValueError(
            f"{root}: expected {expected_count} payload decision files, found {len(paths)}"
        )
    for path in (paths[0], paths[len(paths) // 2], paths[-1]):
        payload = _load_json(path)
        rows = payload.get("per_step") or payload.get("decisions") or []
        if len(rows) != steps:
            raise ValueError(f"{path}: payload decision step count mismatch")
        if any(bool(item.get("schedule_locked")) != locked for item in rows):
            raise ValueError(f"{path}: schedule_locked mismatch")


def validate_generation(row: Mapping[str, Any]) -> dict[str, Any]:
    root = output_path(row)
    expected = int(row["n_prompts"])
    if image_count(root) != expected:
        raise ValueError(f"{root}: generated image coverage mismatch")
    timing = _load_json(root / "timing.json")
    mode = str(row["mode"])
    steps = int(row["num_steps"])
    if timing.get("mode_raw") != mode:
        raise ValueError(f"{root}: mode_raw mismatch")
    if int(timing.get("base_seed", -1)) != 42 or int(timing.get("num_steps", -1)) != steps:
        raise ValueError(f"{root}: seed/steps mismatch")
    if int(timing.get("n_images", -1)) != expected:
        raise ValueError(f"{root}: timing image count mismatch")

    interval = int(row["interval"])
    order = int(row["max_order"])
    if interval and int(timing.get("interval", -1)) != interval:
        raise ValueError(f"{root}: interval mismatch")
    if order and int(timing.get("max_order", -1)) != order:
        raise ValueError(f"{root}: max_order mismatch")
    _same_number(timing.get("hicache_sigma"), row["sigma"], f"{root}: hicache_sigma")
    if _optional(row["first_enhance"]) and mode != "original":
        if int(timing.get("first_enhance", -1)) != int(row["first_enhance"]):
            raise ValueError(f"{root}: first_enhance mismatch")
    _same_number(timing.get("threshold"), row["threshold"], f"{root}: threshold")

    sen_k = _optional(row["sen_k"])
    if sen_k and int(sen_k) and int(timing.get("sencache_K", -1)) != int(sen_k):
        raise ValueError(f"{root}: SenCache K mismatch")
    if mode == "TeaCache" and timing.get("teacache_backbone") != "flux":
        raise ValueError(f"{root}: TeaCache backbone is not flux")
    if mode == "SenCache":
        table = _path(row["sencache_table"])
        if _path(timing.get("sencache_sensitivity_path", "")) != table:
            raise ValueError(f"{root}: SenCache sensitivity path mismatch")
        if timing.get("sencache_sensitivity_sha256") != _sha256(table):
            raise ValueError(f"{root}: SenCache sensitivity hash mismatch")
        metadata = timing.get("sencache_sensitivity_metadata") or {}
        if (
            metadata.get("aggregation") != "q90"
            or int(metadata.get("prompt_count", -1)) != 512
            or int(metadata.get("num_steps", -1)) != 50
            or int(metadata.get("seed", -1)) != 42
        ):
            raise ValueError(f"{root}: SenCache q90 metadata mismatch")
        _same_number(timing.get("sencache_thresh_start"), 0.005, f"{root}: SenCache start")
        _same_number(timing.get("sencache_thresh_main"), row["threshold"], f"{root}: SenCache main")
        _same_number(timing.get("sencache_switch_ratio"), 0.2, f"{root}: SenCache switch")
        if timing.get("sencache_threshold_scale") != "auto":
            raise ValueError(f"{root}: SenCache threshold scale mismatch")
        if int(timing.get("sencache_ret_steps", -1)) != 0:
            raise ValueError(f"{root}: SenCache ret steps mismatch")
        if int(timing.get("sencache_cutoff_steps", 0)) != -1:
            raise ValueError(f"{root}: SenCache cutoff mismatch")
    if mode == "L2P":
        weights = _path(row["l2p_weights"])
        if _path(timing.get("l2p_weights", "")) != weights:
            raise ValueError(f"{root}: L2P weights path mismatch")
        if timing.get("l2p_weight_sha256") != _sha256(weights):
            raise ValueError(f"{root}: L2P weights hash mismatch")
        if timing.get("l2p_target") != "final_hidden":
            raise ValueError(f"{root}: L2P target mismatch")
        if timing.get("l2p_granularity") != "final_hidden":
            raise ValueError(f"{root}: L2P granularity mismatch")
        _same_number(timing.get("l2p_min_abs_weight"), 0.0, f"{root}: L2P min_abs")
    if mode == "SeaCachePayload":
        if timing.get("payload_mode") != "reuse":
            raise ValueError(f"{root}: payload mode is not reuse")
        if timing.get("payload_gate_mode") != "seacache":
            raise ValueError(f"{root}: payload gate is not SeaCache")
        _same_number(timing.get("payload_blend"), 1.0, f"{root}: payload blend")
        _same_number(timing.get("payload_sigma"), 0.5, f"{root}: payload sigma")
        if timing.get("allow_online_payload_schedule_drift") not in (False, None):
            raise ValueError(f"{root}: online schedule drift is enabled")
        schedule = _optional(row["schedule_dir"])
        actual_schedule = timing.get("payload_schedule_dir")
        if schedule:
            if not actual_schedule or _path(actual_schedule) != _path(schedule):
                raise ValueError(f"{root}: fixed schedule path mismatch")
        elif actual_schedule not in (None, ""):
            raise ValueError(f"{root}: native row unexpectedly uses a fixed schedule")
        _validate_payload_decision_sample(
            root, expected_count=expected, steps=steps, locked=bool(schedule)
        )
    return timing


def validate_metrics(
    row: Mapping[str, Any], *, timing: dict[str, Any] | None = None
) -> tuple[dict[str, Any], float | None]:
    root = output_path(row)
    expected = int(row["n_prompts"])
    timing = timing or validate_generation(row)
    validate_original(row)
    metrics = _load_json(root / "metrics.json")
    original = _path(row["original_dir"])
    prompt = _path(row["prompt_file"])
    if _path(metrics.get("acc", "")) != root:
        raise ValueError(f"{root}: metrics acc mismatch")
    if _path(metrics.get("gt", "")) != original:
        raise ValueError(f"{root}: metrics GT mismatch")
    if _path(metrics.get("prompts", "")) != prompt:
        raise ValueError(f"{root}: metrics prompt mismatch")
    if int(metrics.get("n_pairs", -1)) != expected:
        raise ValueError(f"{root}: metric pair count mismatch")
    if metrics.get("indices") != list(range(expected)):
        raise ValueError(f"{root}: metric index coverage mismatch")
    args = metrics.get("args") or {}
    if args:
        if _path(args.get("gt", "")) != original or _path(args.get("prompts", "")) != prompt:
            raise ValueError(f"{root}: metric arguments mismatch")
        if args.get("limit") is not None and int(args["limit"]) != expected:
            raise ValueError(f"{root}: metric limit mismatch")
    for name in METRICS:
        values = (metrics.get("per_image") or {}).get(name)
        summary = (metrics.get("summary") or {}).get(name)
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"{root}: incomplete {name} per-image values")
        if not isinstance(summary, dict) or int(summary.get("n", -1)) != expected:
            raise ValueError(f"{root}: incomplete {name} summary")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{root}: non-finite {name} values")

    ratios = [
        float(item["cached_ratio"])
        for item in timing.get("per_image", [])
        if item.get("cached_ratio") is not None
    ]
    ratio: float | None = sum(ratios) / len(ratios) if ratios else None
    if ratio is None and int(row["interval"]) in INTERVAL_CACHED_STEPS:
        ratio = INTERVAL_CACHED_STEPS[int(row["interval"])] / 50.0
    return metrics, ratio


def validate_row(
    row: Mapping[str, Any], *, require_metrics: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None, float | None]:
    timing = validate_generation(row)
    if not require_metrics:
        return timing, None, None
    metrics, ratio = validate_metrics(row, timing=timing)
    return timing, metrics, ratio


def read_matrix_row(matrix: Path, cell: str) -> dict[str, str]:
    with matrix.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["cell"] == cell]
    if len(rows) != 1:
        raise ValueError(f"{matrix}: expected one row for {cell!r}, found {len(rows)}")
    return rows[0]


def validate_p5_schedule(
    *,
    schedule_dir: Path,
    p5a_matrix: Path,
    expected_prompt: Path,
    expected_original: Path,
    expected_method: str,
    expected_threshold: float,
    expected_first_enhance: int,
    expected_prompts: int = 10000,
    expected_steps: int = 50,
    expected_cached: int = 41,
) -> str:
    row = read_matrix_row(p5a_matrix, "p5:ddb_clean10k:082:native_sea")
    source_run = output_path(row)
    if row["method"] != expected_method or row["mode"] != "SeaCachePayload":
        raise ValueError(f"{p5a_matrix}: native Sea method/mode mismatch")
    _same_number(row["threshold"], expected_threshold, f"{p5a_matrix}: native Sea threshold")
    if int(row["first_enhance"]) != expected_first_enhance:
        raise ValueError(f"{p5a_matrix}: native Sea first_enhance mismatch")
    if _path(row["prompt_file"]) != expected_prompt.resolve():
        raise ValueError(f"{p5a_matrix}: native Sea prompt dataset mismatch")
    if _path(row["original_dir"]) != expected_original.resolve():
        raise ValueError(f"{p5a_matrix}: native Sea original mismatch")
    if int(row["n_prompts"]) != expected_prompts or int(row["num_steps"]) != expected_steps:
        raise ValueError(f"{p5a_matrix}: native Sea prompt/step count mismatch")
    if _optional(row["schedule_dir"]):
        raise ValueError(f"{p5a_matrix}: native Sea source is schedule-locked")
    validate_generation(row)

    root = schedule_dir.resolve()
    expected_names = {f"decisions_{idx:05d}.json" for idx in range(expected_prompts)}
    actual_names = {path.name for path in root.glob("decisions_*.json") if path.is_file()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)[:5]
        extra = sorted(actual_names - expected_names)[:5]
        raise ValueError(
            f"{root}: fixed schedule coverage mismatch; seen={len(actual_names)} "
            f"expected={expected_prompts} missing={missing} extra={extra}"
        )
    manifest = _load_json(root.parent / "schedule_manifest.json")
    if _path(manifest.get("source_decision_run", "")) != source_run:
        raise ValueError(f"{root.parent}: source_decision_run does not match P5a native run")
    if int(manifest.get("source_prompt_count", -1)) != expected_prompts:
        raise ValueError(f"{root.parent}: source prompt count mismatch")
    if int(manifest.get("num_steps", -1)) != expected_steps:
        raise ValueError(f"{root.parent}: source step count mismatch")
    records = [record for record in manifest.get("variants", []) if _path(record.get("dir", "")) == root]
    if len(records) != 1:
        raise ValueError(f"{root.parent}: expected exactly one variant record for {root}")
    variant = records[0].get("variant")
    if variant not in ("majority", "topk_41"):
        raise ValueError(f"{root}: expected majority or topk_41, found {variant!r}")
    if int(records[0].get("n_cached", -1)) != expected_cached:
        raise ValueError(f"{root}: manifest cached-count mismatch")

    reference_bits = None
    for prompt_idx in range(expected_prompts):
        path = root / f"decisions_{prompt_idx:05d}.json"
        payload = _load_json(path)
        if payload.get("prompt_idx") not in (None, prompt_idx):
            raise ValueError(f"{path}: prompt_idx mismatch")
        if payload.get("payload_mode") not in (None, "reuse"):
            raise ValueError(f"{path}: payload mode mismatch")
        if payload.get("payload_gate_mode") not in (None, "seacache"):
            raise ValueError(f"{path}: payload gate mismatch")
        if payload.get("synthetic_schedule") is not True:
            raise ValueError(f"{path}: expected a synthetic fixed schedule")
        if payload.get("synthetic_schedule_variant") != variant:
            raise ValueError(f"{path}: schedule variant disagrees with manifest")
        if _path(payload.get("source_decision_run", "")) != source_run:
            raise ValueError(f"{path}: source_decision_run does not match P5a native run")
        rows = payload.get("per_step") or payload.get("decisions") or []
        steps = [item.get("step") for item in rows if isinstance(item, dict)]
        actions = [item.get("u") for item in rows if isinstance(item, dict)]
        if steps != list(range(expected_steps)) or len(actions) != expected_steps:
            raise ValueError(f"{path}: expected ordered steps 0..{expected_steps - 1}")
        if any(action not in (0, 1) for action in actions):
            raise ValueError(f"{path}: non-binary schedule action")
        if sum(actions) != expected_cached:
            raise ValueError(f"{path}: cached-count mismatch")
        bits = "".join(str(action) for action in actions)
        if payload.get("action_bitstring") not in (None, bits):
            raise ValueError(f"{path}: action_bitstring mismatch")
        if reference_bits is None:
            reference_bits = bits
        elif bits != reference_bits:
            raise ValueError(f"{path}: prompt-specific schedule differs from fixed reference")
    assert reference_bits is not None
    return reference_bits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("generation", "metrics", "original"):
        child = subparsers.add_parser(action)
        child.add_argument("--matrix", type=Path, required=True)
        child.add_argument("--cell", required=True)
    schedule = subparsers.add_parser("p5-schedule")
    schedule.add_argument("--schedule-dir", type=Path, required=True)
    schedule.add_argument("--p5a-matrix", type=Path, required=True)
    schedule.add_argument("--expected-prompt", type=Path, required=True)
    schedule.add_argument("--expected-original", type=Path, required=True)
    schedule.add_argument("--expected-method", required=True)
    schedule.add_argument("--expected-threshold", type=float, required=True)
    schedule.add_argument("--expected-first-enhance", type=int, required=True)
    schedule.add_argument("--expected-prompts", type=int, default=10000)
    schedule.add_argument("--expected-steps", type=int, default=50)
    schedule.add_argument("--expected-cached", type=int, default=41)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.action == "p5-schedule":
            print(validate_p5_schedule(
                schedule_dir=args.schedule_dir,
                p5a_matrix=args.p5a_matrix,
                expected_prompt=args.expected_prompt,
                expected_original=args.expected_original,
                expected_method=args.expected_method,
                expected_threshold=args.expected_threshold,
                expected_first_enhance=args.expected_first_enhance,
                expected_prompts=args.expected_prompts,
                expected_steps=args.expected_steps,
                expected_cached=args.expected_cached,
            ))
            return 0
        row = read_matrix_row(args.matrix, args.cell)
        if args.action == "generation":
            validate_generation(row)
        elif args.action == "metrics":
            validate_row(row, require_metrics=True)
        else:
            validate_original(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
