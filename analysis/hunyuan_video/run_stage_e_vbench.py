#!/usr/bin/env python3
"""Run one official VBench dimension group for one Stage E row."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.hunyuan_video.build_stage_e_vbench_assignments import (
    SEEDS,
    load_vbench_metadata,
)
from analysis.hunyuan_video.stage_e_vbench_staging import (
    validate_vbench_row_tasks,
    validate_staging_payload,
)
from analysis.hunyuan_video.vbench_groups import DIMENSION_GROUPS
from hunyuan_video.config import TaskManifest
from hunyuan_video.records import load_self_hashed_json, sha256_file


VBENCH_COMMIT = "45e79ec14e69a2187202c675d2dbce1a71843d53"
SOURCE_FILTERED_DETAIL_DIMENSIONS = frozenset({"color"})


def expected_names_by_dimension(staging: dict[str, Any], vbench: dict[str, Any]) -> dict[str, frozenset[str]]:
    metadata_rows = vbench.get("metadata_rows")
    if not isinstance(metadata_rows, list) or len(metadata_rows) != 946:
        raise ValueError("VBench metadata rows are incomplete")
    metadata_by_index: dict[int, dict[str, Any]] = {}
    for row in metadata_rows:
        index = row.get("row_index") if isinstance(row, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or index in metadata_by_index:
            raise ValueError("VBench metadata row index drift")
        metadata_by_index[index] = row
    if set(metadata_by_index) != set(range(946)):
        raise ValueError("VBench metadata row coverage drift")

    name_by_pair = {(item["artifact_id"], item["seed"]): item["name"] for item in staging["items"]}
    expected: dict[str, set[str]] = defaultdict(set)
    for artifact in vbench["artifacts"]:
        artifact_id = artifact["artifact_id"]
        indices = artifact.get("metadata_row_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"VBench artifact has no metadata rows: {artifact_id}")
        dimensions: set[str] = set()
        for index in indices:
            row = metadata_by_index.get(index)
            if row is None or row.get("artifact_id") != artifact_id:
                raise ValueError(f"VBench artifact/metadata row drift: {artifact_id}")
            metadata = row.get("metadata")
            raw_dimensions = metadata.get("dimension") if isinstance(metadata, dict) else None
            if (
                not isinstance(raw_dimensions, list)
                or not raw_dimensions
                or any(not isinstance(value, str) for value in raw_dimensions)
            ):
                raise ValueError(f"VBench artifact has no dimensions: {artifact_id}")
            dimensions.update(value.replace(" ", "_") for value in raw_dimensions)
        for seed in SEEDS:
            name = name_by_pair.get((artifact_id, seed))
            if name is None:
                raise ValueError(f"VBench staging omits artifact/seed: {artifact_id}/{seed}")
            for dimension in dimensions:
                expected[dimension].add(name)
    official_dimensions = {dimension for dimensions in DIMENSION_GROUPS.values() for dimension in dimensions}
    if set(expected) != official_dimensions:
        raise ValueError("VBench metadata dimension coverage drift")
    return {dimension: frozenset(names) for dimension, names in expected.items()}


def validate_group_result(
    path: Path,
    *,
    group: str,
    staging: dict[str, Any],
    vbench: dict[str, Any],
) -> dict[str, Any]:
    if group not in DIMENSION_GROUPS:
        raise ValueError(f"unknown official VBench group: {group}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_dimensions = set(DIMENSION_GROUPS[group])
    if not isinstance(payload, dict) or set(payload) != expected_dimensions:
        raise ValueError(f"VBench group dimension coverage mismatch: {group}")
    expected_names = expected_names_by_dimension(staging, vbench)
    staging_dir = Path(staging["staging_dir"]).resolve()
    for dimension in DIMENSION_GROUPS[group]:
        value = payload[dimension]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or not isinstance(value[0], (int, float))
            or not math.isfinite(float(value[0]))
            or not isinstance(value[1], list)
            or not value[1]
        ):
            raise ValueError(f"invalid official VBench result shape: {dimension}")
        detail_names: set[str] = set()
        source_filtered_values: list[float] = []
        for detail in value[1]:
            if not isinstance(detail, dict) or not isinstance(detail.get("video_path"), str):
                raise ValueError(f"VBench detail lacks video_path: {dimension}")
            detail_path = Path(detail["video_path"])
            name = detail_path.name
            # Resolve the parent only: resolving the filename would follow the
            # staging symlink into the generation tree.
            if detail_path.parent.resolve() != staging_dir:
                raise ValueError(f"VBench detail path escapes staging: {dimension}/{name}")
            if name in detail_names:
                raise ValueError(f"duplicate VBench detail path: {dimension}/{name}")
            detail_names.add(name)
            if dimension in SOURCE_FILTERED_DETAIL_DIMENSIONS:
                detail_value = detail.get("cur_success_frame_rate")
                if (
                    isinstance(detail_value, bool)
                    or not isinstance(detail_value, (int, float))
                    or not math.isfinite(float(detail_value))
                    or not 0.0 <= float(detail_value) <= 1.0
                ):
                    raise ValueError(f"invalid source-filtered VBench detail: {dimension}/{name}")
                source_filtered_values.append(float(detail_value))
        expected_detail_names = expected_names[dimension]
        if (
            detail_names - expected_detail_names
            or (
                dimension not in SOURCE_FILTERED_DETAIL_DIMENSIONS
                and detail_names != expected_detail_names
            )
        ):
            raise ValueError(f"VBench detail coverage mismatch: {dimension}")
        if source_filtered_values and not math.isclose(
            float(value[0]),
            sum(source_filtered_values) / len(source_filtered_values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"VBench detail reduction mismatch: {dimension}")
    return payload


def run_group(
    *,
    task_manifest_path: Path,
    row_id: str,
    group: str,
    staging_dir: Path,
    staging_manifest_path: Path,
    output_dir: Path,
    full_info_path: Path,
) -> dict[str, Any]:
    manifest = TaskManifest.load(task_manifest_path)
    vbench = load_vbench_metadata()
    validate_vbench_row_tasks(manifest, row_id, vbench_metadata=vbench)
    if group not in DIMENSION_GROUPS:
        raise ValueError(f"unknown official VBench group: {group}")
    if sha256_file(full_info_path) != vbench.get("source", {}).get("file_sha256"):
        raise ValueError("official VBench full-info source drift")

    staging = load_self_hashed_json(staging_manifest_path, "manifest_sha256")
    validate_staging_payload(
        staging,
        manifest=manifest,
        row_id=row_id,
        staging_dir=staging_dir,
        require_links=True,
    )
    if staging.get("vbench_metadata_sha256") != vbench["manifest_sha256"]:
        raise ValueError("Stage E staging/VBench metadata binding mismatch")
    expected_names = {item["name"] for item in staging["items"]}
    observed_names = {path.name for path in staging_dir.iterdir()}
    if observed_names != expected_names:
        raise ValueError("official VBench staging filename coverage mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row_id}_{group}_eval_results.json"
    if output_path.exists():
        validate_group_result(
            output_path,
            group=group,
            staging=staging,
            vbench=vbench,
        )
        return {
            "group": group,
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "row_id": row_id,
            "skipped": True,
        }

    import torch
    from vbench import VBench

    evaluator = VBench(torch.device("cuda"), str(full_info_path), str(output_dir))
    evaluator.evaluate(
        videos_path=str(staging_dir),
        name=f"{row_id}_{group}",
        dimension_list=list(DIMENSION_GROUPS[group]),
        local=True,
        mode="vbench_standard",
        read_frame=False,
        imaging_quality_preprocessing_mode="longer",
    )
    validate_group_result(
        output_path,
        group=group,
        staging=staging,
        vbench=vbench,
    )
    return {
        "group": group,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_id": row_id,
        "skipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--group", choices=tuple(DIMENSION_GROUPS), required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-info", type=Path, required=True)
    args = parser.parse_args()
    result = run_group(
        task_manifest_path=args.task_manifest,
        row_id=args.row_id,
        group=args.group,
        staging_dir=args.staging_dir,
        staging_manifest_path=args.staging_manifest,
        output_dir=args.output_dir,
        full_info_path=args.full_info,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
