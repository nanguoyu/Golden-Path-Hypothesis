#!/usr/bin/env python3
"""Stage one Stage E row for official VBench and remove it explicitly."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.hunyuan_video.build_stage_e_vbench_assignments import (
    PROTOCOL_ID,
    SEEDS,
    TASKS_PER_ROW,
    load_vbench_metadata,
)
from analysis.hunyuan_video.vbench_groups import DIMENSION_GROUPS
from hunyuan_video.config import TaskManifest, hash_json, load_protocol, require_sha256
from hunyuan_video.records import (
    load_self_hashed_json,
    sha256_file,
    write_immutable_json,
)


SCHEMA = "hunyuan_video.stage_e_vbench_staging.v1"
SUMMARY_SCHEMA = "hunyuan_video.stage_e_vbench_summary.v1"
SEED_TO_INDEX = {seed: index for index, seed in enumerate(SEEDS)}
VBENCH_AUDIT_SAMPLE_PATH = ROOT / "resources/hunyuan_video/vbench_audit_sample.v1.json"
SUPPORTED_STAGES = frozenset({"E6", "E7"})


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _row(manifest: TaskManifest, row_id: str) -> Any:
    matches = [
        row
        for row in manifest.formal_rows
        if row.row_id == row_id and row.stage in SUPPORTED_STAGES and row.split == "VBench"
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage E VBench row identity mismatch: {row_id}")
    return matches[0]


def validate_vbench_row_tasks(
    manifest: TaskManifest,
    row_id: str,
    *,
    vbench_metadata: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Validate the exact artifact/seed coverage needed to stage one row."""

    row = _row(manifest, row_id)
    if row.protocol_id != PROTOCOL_ID or row.seeds != SEEDS or row.repeats != 1 or row.source_protocol_id is not None:
        raise ValueError(f"Stage E VBench row protocol drift: {row_id}")
    vbench = vbench_metadata or load_vbench_metadata()
    artifacts = vbench["artifacts"]
    artifact_by_id = {item["artifact_id"]: item for item in artifacts}
    expected_pairs = {(artifact_id, seed) for artifact_id in artifact_by_id for seed in SEEDS}
    tasks = tuple(
        sorted(
            (task for task in manifest.tasks if task.formal_row_id == row_id),
            key=lambda task: (
                task.run_spec.seed,
                task.run_spec.prompt_id,
                task.run_spec.task_id,
            ),
        )
    )
    if len(tasks) != TASKS_PER_ROW:
        raise ValueError(f"Stage E VBench row task coverage drift: {row_id}")
    task_ids: set[str] = set()
    observed_pairs: set[tuple[str, int]] = set()
    for task in tasks:
        run = task.run_spec
        artifact = artifact_by_id.get(run.prompt_id)
        if (
            run.task_id in task_ids
            or artifact is None
            or run.prompt != artifact.get("prompt")
            or run.phase != row.stage
            or run.protocol_id != row.protocol_id
            or run.mode != row.mode
            or run.seed not in SEEDS
            or run.repeat != 0
            or run.method_config != row.method_config
        ):
            raise ValueError(f"Stage E VBench task identity drift: {run.task_id}")
        task_ids.add(run.task_id)
        observed_pairs.add((run.prompt_id, run.seed))
    if observed_pairs != expected_pairs:
        raise ValueError(f"Stage E VBench artifact/seed coverage drift: {row_id}")
    return tasks


def _safe_filename(prompt: str, seed: int) -> str:
    if seed not in SEED_TO_INDEX:
        raise ValueError(f"unexpected Stage E VBench seed: {seed}")
    name = f"{prompt}-{SEED_TO_INDEX[seed]}.mp4"
    if Path(name).name != name or "\n" in name or "\x00" in name:
        raise ValueError(f"unsafe VBench filename: {name!r}")
    if len(os.fsencode(name)) > 255:
        raise ValueError(f"overlong VBench filename: {name!r}")
    return name


def validate_staging_payload(
    payload: dict[str, Any],
    *,
    manifest: TaskManifest,
    row_id: str,
    staging_dir: Path,
    require_links: bool,
) -> dict[str, Any]:
    if (
        payload.get("schema") != SCHEMA
        or payload.get("row_id") != row_id
        or payload.get("task_manifest_sha256") != manifest.sha256
        or payload.get("item_count") != TASKS_PER_ROW
        or Path(payload.get("staging_dir", "")).resolve() != staging_dir.resolve()
    ):
        raise ValueError("Stage E VBench staging identity mismatch")
    row = _row(manifest, row_id)
    if payload.get("row_sha256") != row.sha256:
        raise ValueError("Stage E VBench staging row binding mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != TASKS_PER_ROW:
        raise ValueError("Stage E VBench staging item coverage mismatch")
    task_ids: set[str] = set()
    names: set[str] = set()
    source_paths: set[str] = set()
    artifact_seed_pairs: set[tuple[str, int]] = set()
    row_tasks = {task.run_spec.task_id: task for task in manifest.tasks if task.formal_row_id == row_id}
    if len(row_tasks) != TASKS_PER_ROW:
        raise ValueError("Stage E VBench row task coverage mismatch")
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Stage E VBench staging item must be an object")
        task_id = item.get("task_id")
        artifact_id = item.get("artifact_id")
        seed = item.get("seed")
        name = item.get("name")
        source_path = item.get("source_path")
        latent_source_path = item.get("latent_source_path")
        latent_sha256 = item.get("latent_sha256")
        if (
            not isinstance(task_id, str)
            or not isinstance(artifact_id, str)
            or isinstance(seed, bool)
            or seed not in SEED_TO_INDEX
            or not isinstance(name, str)
            or not isinstance(source_path, str)
            or not isinstance(item.get("candidate_sha256"), str)
        ):
            raise ValueError("invalid Stage E VBench staging item identity")
        require_sha256(item["candidate_sha256"], "candidate_sha256")
        if (latent_source_path is None) != (latent_sha256 is None):
            raise ValueError("Stage E VBench latent metadata must be present as a pair")
        if latent_source_path is not None:
            if not isinstance(latent_source_path, str) or not isinstance(latent_sha256, str):
                raise ValueError("invalid Stage E VBench latent metadata")
            require_sha256(latent_sha256, "latent_sha256")
        task = row_tasks.get(task_id)
        if task is None:
            raise ValueError(f"Stage E VBench staging references an unknown task: {task_id}")
        run = task.run_spec
        if (
            artifact_id != run.prompt_id
            or seed != run.seed
            or name != _safe_filename(run.prompt, run.seed)
            or not Path(source_path).is_absolute()
            or (latent_source_path is not None and not Path(latent_source_path).is_absolute())
        ):
            raise ValueError(f"Stage E VBench staging/task identity drift: {task_id}")
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError(f"unsafe VBench staging name: {name!r}")
        if (
            task_id in task_ids
            or name in names
            or source_path in source_paths
            or (latent_source_path is not None and latent_source_path in source_paths)
            or (artifact_id, seed) in artifact_seed_pairs
        ):
            raise ValueError("duplicate Stage E VBench staging identity")
        task_ids.add(task_id)
        names.add(name)
        source_paths.add(source_path)
        if latent_source_path is not None:
            source_paths.add(latent_source_path)
        artifact_seed_pairs.add((artifact_id, seed))
        path = staging_dir / name
        if path.parent.resolve() != staging_dir.resolve():
            raise ValueError(f"VBench staging path escapes its root: {path}")
        if require_links:
            if (
                not path.is_symlink()
                or str(path.resolve()) != source_path
                or not Path(source_path).is_file()
                or (latent_source_path is not None and not Path(latent_source_path).is_file())
            ):
                raise ValueError(f"Stage E VBench staging target drift: {path}")
    if task_ids != set(row_tasks):
        raise ValueError("Stage E VBench staging/task coverage mismatch")
    return payload


def build_staging(
    *,
    task_manifest_path: Path,
    generation_root: Path,
    row_id: str,
    staging_dir: Path,
    staging_manifest_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    from evaluation.eval_hunyuan_c2 import _resolve_accepted_attempt

    manifest = TaskManifest.load(task_manifest_path)
    vbench = load_vbench_metadata()
    tasks = validate_vbench_row_tasks(
        manifest,
        row_id,
        vbench_metadata=vbench,
    )
    row = _row(manifest, row_id)
    protocol = load_protocol(row.protocol_id)
    if manifest.protocol_manifest_sha256s[row.protocol_id] != protocol.manifest_sha256:
        raise ValueError("Stage E VBench protocol binding drift")
    if workers < 1:
        raise ValueError("workers must be positive")

    if staging_manifest_path.exists():
        existing = load_self_hashed_json(staging_manifest_path, "manifest_sha256")
        return validate_staging_payload(
            existing,
            manifest=manifest,
            row_id=row_id,
            staging_dir=staging_dir,
            require_links=True,
        )

    prepared: list[tuple[Any, str]] = []
    names: set[str] = set()
    for task in tasks:
        run = task.run_spec
        name = _safe_filename(run.prompt, run.seed)
        if name in names:
            raise ValueError(f"duplicate official VBench filename: {name!r}")
        names.add(name)
        prepared.append((task, name))
    if len(prepared) != TASKS_PER_ROW:
        raise AssertionError("Stage E VBench prepared-task coverage drift")

    def resolve(
        item: tuple[Any, str],
    ) -> tuple[
        Any,
        str,
        dict[str, Any],
        Path,
        dict[str, Any] | None,
        Path | None,
    ]:
        task, name = item
        run = task.run_spec
        accepted = _resolve_accepted_attempt(
            generation_root,
            manifest,
            row,
            task,
            protocol.protocol_sha256,
        )
        video = accepted["artifacts"]["video"]
        latent = accepted["artifacts"].get("final_latent")
        if latent is not None and not isinstance(latent, dict):
            raise ValueError(f"invalid final latent metadata: {run.task_id}")
        source = (accepted["attempt_dir"] / video["relative_path"]).resolve()
        latent_source = (
            (accepted["attempt_dir"] / latent["relative_path"]).resolve() if latent is not None else None
        )
        return task, name, video, source, latent, latent_source

    if workers == 1:
        resolved = list(map(resolve, prepared))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            resolved = list(executor.map(resolve, prepared))

    staging_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for task, name, video, source, latent, latent_source in resolved:
        destination = staging_dir / name
        try:
            destination.symlink_to(source)
        except FileExistsError:
            if not destination.is_symlink() or destination.resolve() != source:
                raise FileExistsError(f"VBench staging collision: {destination}")
        run = task.run_spec
        staged_item = {
            "artifact_id": run.prompt_id,
            "candidate_sha256": video["sha256"],
            "name": name,
            "seed": run.seed,
            "source_path": str(source),
            "task_id": run.task_id,
        }
        if latent is not None and latent_source is not None:
            staged_item.update(
                {
                    "latent_sha256": latent["sha256"],
                    "latent_source_path": str(latent_source),
                }
            )
        items.append(staged_item)
    payload = {
        "schema": SCHEMA,
        "item_count": len(items),
        "items": items,
        "protocol_id": row.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "row_id": row_id,
        "row_sha256": row.sha256,
        "staging_dir": str(staging_dir.resolve()),
        "task_manifest_sha256": manifest.sha256,
        "vbench_metadata_sha256": vbench["manifest_sha256"],
    }
    payload["manifest_sha256"] = hash_json(payload)
    validate_staging_payload(
        payload,
        manifest=manifest,
        row_id=row_id,
        staging_dir=staging_dir,
        require_links=True,
    )
    write_immutable_json(staging_manifest_path, payload)
    return payload


def _load_remove_allowlist(path: Path, item_by_task: dict[str, dict[str, Any]]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
        raise TypeError("raw-video allowlist must be a JSON list of task IDs")
    if len(set(payload)) != len(payload):
        raise ValueError("raw-video allowlist contains duplicate task IDs")
    unknown = sorted(set(payload) - set(item_by_task))
    if unknown:
        raise ValueError(f"raw-video allowlist contains unknown task IDs: {unknown[:3]}")
    return payload


def load_vbench_audit_artifact_ids(
    path: Path = VBENCH_AUDIT_SAMPLE_PATH,
) -> tuple[str, ...]:
    payload = load_self_hashed_json(path, "manifest_sha256")
    items = payload.get("items")
    dimension_to_artifact = payload.get("dimension_to_artifact")
    if (
        payload.get("schema") != "hunyuan_video.vbench_audit_sample.v1"
        or payload.get("algorithm") != "dimension-lexicographic-unique-v1"
        or payload.get("artifact_count") != 16
        or payload.get("coverage_count") != 16
        or payload.get("seeds") != list(SEEDS)
        or not isinstance(items, list)
        or len(items) != 16
        or not isinstance(dimension_to_artifact, dict)
        or len(dimension_to_artifact) != 16
    ):
        raise ValueError("frozen VBench audit sample identity mismatch")
    artifact_ids = tuple(item.get("artifact_id") for item in items if isinstance(item, dict))
    if (
        len(artifact_ids) != 16
        or any(not isinstance(value, str) or not value for value in artifact_ids)
        or len(set(artifact_ids)) != 16
        or hash_json(list(artifact_ids)) != payload.get("artifact_ids_sha256")
        or set(dimension_to_artifact.values()) != set(artifact_ids)
    ):
        raise ValueError("frozen VBench audit sample artifact coverage mismatch")
    return artifact_ids


def _cleanup_context(
    *,
    task_manifest_path: Path,
    staging_manifest_path: Path,
    summary_path: Path,
) -> tuple[TaskManifest, dict[str, Any], Path]:
    from analysis.hunyuan_video.run_stage_e_vbench import validate_group_result

    manifest = TaskManifest.load(task_manifest_path)
    staging = load_self_hashed_json(staging_manifest_path, "manifest_sha256")
    summary = load_self_hashed_json(summary_path, "manifest_sha256")
    row_id = staging.get("row_id")
    if not isinstance(row_id, str):
        raise ValueError("Stage E VBench staging row ID is invalid")
    validate_vbench_row_tasks(manifest, row_id)
    staging_dir = Path(staging.get("staging_dir", ""))
    validate_staging_payload(
        staging,
        manifest=manifest,
        row_id=row_id,
        staging_dir=staging_dir,
        require_links=False,
    )
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("row_id") != row_id
        or summary.get("row_sha256") != staging.get("row_sha256")
        or summary.get("task_manifest_sha256") != manifest.sha256
        or summary.get("staging_manifest_sha256") != staging["manifest_sha256"]
        or summary.get("vbench_metadata_sha256") != staging.get("vbench_metadata_sha256")
        or summary.get("dimension_count") != 16
        or summary.get("group_count") != 4
    ):
        raise ValueError("Stage E VBench cleanup requires a complete bound summary")
    expected_dimensions = {dimension for dimensions in DIMENSION_GROUPS.values() for dimension in dimensions}
    dimension_scores = summary.get("dimension_scores")
    detail_counts = summary.get("detail_counts")
    detail_hashes = summary.get("detail_name_sha256s")
    group_files = summary.get("group_files")
    final_scores = [
        summary.get("quality_score"),
        summary.get("semantic_score"),
        summary.get("total_score"),
    ]
    if (
        not isinstance(dimension_scores, dict)
        or set(dimension_scores) != expected_dimensions
        or not isinstance(detail_counts, dict)
        or set(detail_counts) != expected_dimensions
        or not isinstance(detail_hashes, dict)
        or set(detail_hashes) != expected_dimensions
        or not isinstance(group_files, list)
        or len(group_files) != 4
        or {item.get("group") for item in group_files if isinstance(item, dict)} != set(DIMENSION_GROUPS)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in [*dimension_scores.values(), *final_scores]
        )
    ):
        raise ValueError("Stage E VBench cleanup summary coverage is incomplete")
    for item in group_files:
        path = Path(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError("Stage E VBench cleanup summary group-file binding drift")
        group = item.get("group")
        payload = validate_group_result(
            path,
            group=group,
            staging=staging,
            vbench=load_vbench_metadata(),
        )
        for dimension in DIMENSION_GROUPS[group]:
            if float(payload[dimension][0]) != float(dimension_scores[dimension]):
                raise ValueError("Stage E VBench cleanup summary differs from group results")
    return manifest, staging, staging_dir


def build_raw_video_allowlist(
    *,
    task_manifest_path: Path,
    staging_manifest_path: Path,
    summary_path: Path,
    retain_artifact_ids_path: Path,
) -> list[str]:
    _, staging, _ = _cleanup_context(
        task_manifest_path=task_manifest_path,
        staging_manifest_path=staging_manifest_path,
        summary_path=summary_path,
    )
    retain = json.loads(retain_artifact_ids_path.read_text(encoding="utf-8"))
    if not isinstance(retain, list) or any(not isinstance(value, str) for value in retain):
        raise TypeError("retained artifact IDs must be a JSON list of strings")
    if len(set(retain)) != len(retain):
        raise ValueError("retained artifact IDs contain duplicates")
    expected_retain = set(load_vbench_audit_artifact_ids())
    if set(retain) != expected_retain:
        raise ValueError("retained artifact IDs differ from the frozen VBench audit sample")
    observed_artifacts = {item["artifact_id"] for item in staging["items"]}
    unknown = sorted(set(retain) - observed_artifacts)
    if unknown:
        raise ValueError(f"retained artifact IDs are unknown: {unknown[:3]}")
    retain_ids = set(retain)
    return sorted(item["task_id"] for item in staging["items"] if item["artifact_id"] not in retain_ids)


def remove_staging(
    *,
    task_manifest_path: Path,
    staging_manifest_path: Path,
    summary_path: Path,
    raw_video_allowlist_path: Path,
    workers: int = 1,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
    _, staging, staging_dir = _cleanup_context(
        task_manifest_path=task_manifest_path,
        staging_manifest_path=staging_manifest_path,
        summary_path=summary_path,
    )

    item_by_task = {item["task_id"]: item for item in staging["items"]}
    delete_task_ids = _load_remove_allowlist(raw_video_allowlist_path, item_by_task)

    def validate_link(item: Mapping[str, Any]) -> None:
        path = staging_dir / item["name"]
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or str(path.resolve()) != item["source_path"]:
                raise ValueError(f"Stage E VBench staging target drift: {path}")

    # Validate all still-present staging links before deleting any source media.
    if workers == 1:
        for item in staging["items"]:
            validate_link(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(validate_link, staging["items"]))

    def verify_sources(task_id: str) -> tuple[list[Path], int]:
        item = item_by_task[task_id]
        artifact_fields = [("source_path", "candidate_sha256")]
        if "latent_source_path" in item or "latent_sha256" in item:
            if not all(key in item for key in ("latent_source_path", "latent_sha256")):
                raise ValueError(
                    "Stage E VBench latent metadata must be present as a pair"
                )
            artifact_fields.append(("latent_source_path", "latent_sha256"))
        present: list[Path] = []
        missing = 0
        for path_key, sha_key in artifact_fields:
            source = Path(item[path_key])
            if not source.exists():
                missing += 1
                continue
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"cleanup target is not a regular file: {source}")
            if sha256_file(source) != item[sha_key]:
                raise ValueError(f"cleanup target hash drift: {source}")
            present.append(source)
        return present, missing

    if workers == 1:
        source_groups = [verify_sources(task_id) for task_id in delete_task_ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            source_groups = list(executor.map(verify_sources, delete_task_ids))
    verified_sources = [source for sources, _ in source_groups for source in sources]
    already_missing = sum(missing for _, missing in source_groups)

    # All allowlisted media are valid before the first destructive operation.
    if workers == 1:
        for source in verified_sources:
            source.unlink()
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(Path.unlink, verified_sources))
    deleted = len(verified_sources)

    def remove_link(item: Mapping[str, Any]) -> int:
        path = staging_dir / item["name"]
        if not path.exists() and not path.is_symlink():
            return 0
        if not path.is_symlink() or str(path.resolve()) != item["source_path"]:
            raise ValueError(f"Stage E VBench staging target drift: {path}")
        path.unlink()
        return 1

    if workers == 1:
        removed_links = sum(remove_link(item) for item in staging["items"])
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            removed_links = sum(executor.map(remove_link, staging["items"]))
    if staging_dir.exists():
        staging_dir.rmdir()
    return {
        "already_missing_artifacts": already_missing,
        "deleted_artifacts": deleted,
        "removed_staging_links": removed_links,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--task-manifest", type=Path, required=True)
    stage.add_argument("--generation-root", type=Path, required=True)
    stage.add_argument("--row-id", required=True)
    stage.add_argument("--staging-dir", type=Path, required=True)
    stage.add_argument("--staging-manifest", type=Path, required=True)
    stage.add_argument("--workers", type=_positive_int, default=1)
    allowlist = subparsers.add_parser("allowlist")
    allowlist.add_argument("--task-manifest", type=Path, required=True)
    allowlist.add_argument("--staging-manifest", type=Path, required=True)
    allowlist.add_argument("--summary", type=Path, required=True)
    allowlist.add_argument("--retain-artifact-ids", type=Path, required=True)
    allowlist.add_argument("--output", type=Path, required=True)
    remove = subparsers.add_parser("remove")
    remove.add_argument("--task-manifest", type=Path, required=True)
    remove.add_argument("--staging-manifest", type=Path, required=True)
    remove.add_argument("--summary", type=Path, required=True)
    remove.add_argument("--raw-video-allowlist", type=Path, required=True)
    remove.add_argument("--workers", type=_positive_int, default=1)
    args = parser.parse_args()
    if args.command == "stage":
        result = build_staging(
            task_manifest_path=args.task_manifest,
            generation_root=args.generation_root,
            row_id=args.row_id,
            staging_dir=args.staging_dir,
            staging_manifest_path=args.staging_manifest,
            workers=args.workers,
        )
        print(json.dumps({"manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    elif args.command == "allowlist":
        result = build_raw_video_allowlist(
            task_manifest_path=args.task_manifest,
            staging_manifest_path=args.staging_manifest,
            summary_path=args.summary,
            retain_artifact_ids_path=args.retain_artifact_ids,
        )
        write_immutable_json(args.output, result)
        print(json.dumps({"output": str(args.output), "task_count": len(result)}, sort_keys=True))
    else:
        result = remove_staging(
            task_manifest_path=args.task_manifest,
            staging_manifest_path=args.staging_manifest,
            summary_path=args.summary,
            raw_video_allowlist_path=args.raw_video_allowlist,
            workers=args.workers,
        )
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
