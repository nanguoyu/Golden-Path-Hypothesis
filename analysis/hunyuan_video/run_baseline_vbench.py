#!/usr/bin/env python3
"""Run the four official VBench dimension groups on one staged baseline-matrix row.

A row is one (mode, budget) of the baseline matrix: three seed cells staged by
`baseline_vbench_staging.py` into ONE directory of `<prompt>-{0,1,2}.mp4` links.
This is the Stage-E `run_stage_e_vbench.py::run_group` call, minus the Stage-E
TaskManifest binding (the matrix has none); the staging manifest and the
frozen VBench metadata are what bind the link set to the prompts VBench scores.

    python analysis/hunyuan_video/run_baseline_vbench.py \\
        --staging-dir $DATA/.../vbench/staging/seacache_K37 \\
        --staging-manifest $DATA/.../vbench/staging/seacache_K37.json \\
        --name seacache_K37 --out-dir $DATA/.../vbench/seacache_K37

Groups whose `<name>_<group>_eval_results.json` already exists are validated
and skipped; `<name>_summary.json` collects the 16 dimension scores.
"""

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

from analysis.hunyuan_video.baseline_vbench_staging import PROMPT_SCHEMA, SCHEMA  # noqa: E402
from analysis.hunyuan_video.build_stage_e_vbench_assignments import load_vbench_metadata  # noqa: E402
from analysis.hunyuan_video.run_stage_e_vbench import (  # noqa: E402
    SOURCE_FILTERED_DETAIL_DIMENSIONS,
    VBENCH_COMMIT,
)
from analysis.hunyuan_video.vbench_groups import DIMENSION_GROUPS  # noqa: E402
from hunyuan_video.records import load_self_hashed_json, sha256_file  # noqa: E402

DEFAULT_PROMPT_MANIFEST = ROOT / "resources/hunyuan_video/evaluation/vbench944.json"
DEFAULT_FULL_INFO = ROOT / "reference/vbench/code/vbench/VBench_full_info.json"
ALL_DIMENSIONS = tuple(d for dims in DIMENSION_GROUPS.values() for d in dims)


def artifact_dimensions(vbench: dict[str, Any]) -> dict[str, frozenset[str]]:
    """artifact_id -> the official dimensions that score it, from the frozen metadata."""
    rows_by_index = {row["row_index"]: row for row in vbench["metadata_rows"]}
    out: dict[str, frozenset[str]] = {}
    for artifact in vbench["artifacts"]:
        dims: set[str] = set()
        for index in artifact["metadata_row_indices"]:
            row = rows_by_index[index]
            if row["artifact_id"] != artifact["artifact_id"]:
                raise ValueError(f"VBench artifact/metadata row drift: {artifact['artifact_id']}")
            dims.update(v.replace(" ", "_") for v in row["metadata"]["dimension"])
        if not dims:
            raise ValueError(f"VBench artifact has no dimensions: {artifact['artifact_id']}")
        out[artifact["artifact_id"]] = frozenset(dims)
    return out


def expected_names_by_dimension(
    staging: dict[str, Any], prompt_ids: list[str], vbench: dict[str, Any]
) -> dict[str, frozenset[str]]:
    """Which staged file names each official dimension should score."""
    dims_by_artifact = artifact_dimensions(vbench)
    expected: dict[str, set[str]] = defaultdict(set)
    for item in staging["items"]:
        artifact_id = prompt_ids[int(item["prompt_idx"])]
        for dim in dims_by_artifact[artifact_id]:
            expected[dim].add(item["name"])
    if set(expected) != set(ALL_DIMENSIONS):
        raise ValueError("VBench dimension coverage drift in staging")
    return {dim: frozenset(names) for dim, names in expected.items()}


def load_staging(
    staging_manifest_path: Path,
    *,
    staging_dir: Path,
    prompt_manifest_path: Path,
    vbench: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, frozenset[str]]]:
    staging = load_self_hashed_json(staging_manifest_path, "manifest_sha256")
    if staging.get("schema") != SCHEMA:
        raise ValueError(f"{staging_manifest_path} is not a baseline VBench staging manifest")
    if Path(staging["staging_dir"]).resolve() != staging_dir.resolve():
        raise ValueError(f"staging manifest describes {staging['staging_dir']}, not {staging_dir}")
    prompt_manifest = load_self_hashed_json(prompt_manifest_path, "manifest_sha256")
    if prompt_manifest.get("schema") != PROMPT_SCHEMA:
        raise ValueError(f"{prompt_manifest_path} is not an evaluation prompt manifest")
    if prompt_manifest.get("dataset") != staging.get("dataset"):
        raise ValueError(f"staging is {staging.get('dataset')!r} but prompt manifest is "
                         f"{prompt_manifest.get('dataset')!r}")
    if len(prompt_manifest["items"]) != staging["prompt_count"]:
        raise ValueError("staging prompt_count does not match the prompt manifest")
    prompt_ids = [str(item["prompt_id"]) for item in prompt_manifest["items"]]
    if set(prompt_ids) != {a["artifact_id"] for a in vbench["artifacts"]}:
        raise ValueError("prompt manifest ids are not the frozen VBench artifact ids")

    manifest_names = {item["name"] for item in staging["items"]}
    if len(manifest_names) != len(staging["items"]) or len(manifest_names) != staging["item_count"]:
        raise ValueError("staging manifest names are not unique or item_count drifted")
    observed = {p.name for p in staging_dir.iterdir()}
    if observed != manifest_names:
        raise ValueError(f"staging dir has {len(observed)} names, manifest has "
                         f"{len(manifest_names)}; they differ")
    expected = expected_names_by_dimension(staging, prompt_ids, vbench)
    if set().union(*expected.values()) != manifest_names:
        raise ValueError("staged names are not exactly the VBench-scored artifacts")
    return staging, expected


def validate_group_result(
    path: Path, *, group: str, staging_dir: Path, expected: dict[str, frozenset[str]]
) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dims = DIMENSION_GROUPS[group]
    if not isinstance(payload, dict) or set(payload) != set(dims):
        raise ValueError(f"VBench group dimension coverage mismatch: {group}")
    scores: dict[str, float] = {}
    staging_dir = staging_dir.resolve()
    for dim in dims:
        value = payload[dim]
        if (not isinstance(value, list) or len(value) != 2 or isinstance(value[0], bool)
                or not isinstance(value[0], (int, float)) or not math.isfinite(float(value[0]))
                or not isinstance(value[1], list) or not value[1]):
            raise ValueError(f"invalid official VBench result shape: {dim}")
        names: set[str] = set()
        for detail in value[1]:
            detail_path = Path(detail["video_path"])
            # resolve the parent only: resolving the name follows the link
            if detail_path.parent.resolve() != staging_dir:
                raise ValueError(f"VBench detail path escapes staging: {dim}/{detail_path.name}")
            if detail_path.name in names:
                raise ValueError(f"duplicate VBench detail path: {dim}/{detail_path.name}")
            names.add(detail_path.name)
        if names - expected[dim] or (
            dim not in SOURCE_FILTERED_DETAIL_DIMENSIONS and names != expected[dim]
        ):
            raise ValueError(f"VBench detail coverage mismatch: {dim}")
        scores[dim] = float(value[0])
    return scores


def run(
    *,
    staging_dir: Path,
    staging_manifest_path: Path,
    name: str,
    out_dir: Path,
    groups: list[str],
    prompt_manifest_path: Path,
    full_info_path: Path,
) -> dict[str, Any]:
    vbench = load_vbench_metadata()
    if sha256_file(full_info_path) != vbench.get("source", {}).get("file_sha256"):
        raise ValueError("official VBench full-info source drift")
    staging, expected = load_staging(
        staging_manifest_path,
        staging_dir=staging_dir,
        prompt_manifest_path=prompt_manifest_path,
        vbench=vbench,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = None
    group_records: dict[str, Any] = {}
    dimension_scores: dict[str, float] = {}
    for group in groups:
        output_path = out_dir / f"{name}_{group}_eval_results.json"
        skipped = output_path.exists()
        if not skipped:
            if evaluator is None:
                import torch
                from vbench import VBench

                evaluator = VBench(torch.device("cuda"), str(full_info_path), str(out_dir))
            evaluator.evaluate(
                videos_path=str(staging_dir),
                name=f"{name}_{group}",
                dimension_list=list(DIMENSION_GROUPS[group]),
                local=True,
                mode="vbench_standard",
                read_frame=False,
                imaging_quality_preprocessing_mode="longer",
            )
        scores = validate_group_result(
            output_path, group=group, staging_dir=staging_dir, expected=expected
        )
        dimension_scores.update(scores)
        group_records[group] = {
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "skipped": skipped,
        }
        print(f"[{'skip' if skipped else 'done'}] {group}: "
              + json.dumps(scores, sort_keys=True), flush=True)

    summary = {
        "schema": "hunyuan_video.baseline_vbench_summary.v1",
        "name": name,
        "dataset": staging["dataset"],
        "cells": staging["cells"],
        "staging_dir": str(staging_dir.resolve()),
        "staging_manifest": str(staging_manifest_path),
        "staging_manifest_sha256": staging["manifest_sha256"],
        "vbench_metadata_sha256": vbench["manifest_sha256"],
        "vbench_commit": VBENCH_COMMIT,
        "groups": group_records,
        "dimensions": {dim: dimension_scores[dim] for dim in ALL_DIMENSIONS
                       if dim in dimension_scores},
    }
    if len(dimension_scores) == len(ALL_DIMENSIONS):
        (out_dir / f"{name}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=tuple(DIMENSION_GROUPS),
                        default=list(DIMENSION_GROUPS))
    parser.add_argument("--prompt-manifest", type=Path, default=DEFAULT_PROMPT_MANIFEST)
    parser.add_argument("--full-info", type=Path, default=DEFAULT_FULL_INFO)
    args = parser.parse_args(argv)
    summary = run(
        staging_dir=args.staging_dir,
        staging_manifest_path=args.staging_manifest,
        name=args.name,
        out_dir=args.out_dir,
        groups=list(args.groups),
        prompt_manifest_path=args.prompt_manifest,
        full_info_path=args.full_info,
    )
    print(json.dumps(summary["dimensions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
