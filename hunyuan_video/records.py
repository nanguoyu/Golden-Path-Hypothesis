from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import canonical_json_bytes, hash_json, require_sha256


ATTEMPT_IDS = (0, 1)
DEFAULT_REQUIRED_ARTIFACT_ROLES = ("video", "final_latent", "action")
AttemptValidator = Callable[[Path, Mapping[str, Any]], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = canonical_json_bytes(payload) + b"\n"
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_immutable_json(path: Path, payload: Any) -> None:
    """Atomically create a JSON record, accepting only byte-identical replays."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable JSON record differs: {path}")
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.immutable.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(f"immutable JSON record differs: {path}")
    finally:
        if tmp.exists():
            tmp.unlink()


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return hash_json({key: value for key, value in payload.items() if key != field})


def load_self_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = require_sha256(payload.get(hash_field, ""), hash_field)
    if _self_hash(payload, hash_field) != expected:
        raise ValueError(f"{path.name} self-hash mismatch")
    return payload


def _inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"artifact path escapes attempt directory: {path}")
    return resolved


def _attempt_dir_name(attempt_id: int) -> str:
    if attempt_id not in ATTEMPT_IDS:
        raise ValueError(f"attempt_id must be one of {ATTEMPT_IDS}")
    return f"attempt_{attempt_id:03d}"


def _open_video_reader(video_path: Path) -> Any:
    import imageio.v2 as imageio

    return imageio.get_reader(str(video_path))


def probe_video_protocol(
    video_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
) -> dict[str, Any]:
    """Stream-decode every frame and validate the frozen protocol geometry."""

    if min(width, height, fps, frames) <= 0:
        raise ValueError("video protocol dimensions, fps, and frames must be positive")
    try:
        reader = _open_video_reader(video_path)
    except Exception as error:
        raise ValueError(f"imageio could not open video: {video_path}") from error
    observed_frames = 0
    observed_fps: Any = None
    try:
        metadata = reader.get_meta_data()
        observed_fps = metadata.get("fps") if isinstance(metadata, Mapping) else None
        if isinstance(observed_fps, bool) or not isinstance(observed_fps, (int, float)):
            raise ValueError(f"video metadata has invalid fps: {observed_fps}")
        if not math.isfinite(float(observed_fps)) or float(observed_fps) != float(fps):
            raise ValueError(f"video fps mismatch: {observed_fps} != {fps}")
        for frame in reader:
            observed_frames += 1
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            if shape != (height, width, 3):
                raise ValueError(
                    f"video frame {observed_frames - 1} shape mismatch: " f"{shape} != {(height, width, 3)}"
                )
            if str(dtype) != "uint8":
                raise ValueError(f"video frame {observed_frames - 1} dtype mismatch: {dtype} != uint8")
            if observed_frames > frames:
                raise ValueError(f"video frame-count exceeds expected {frames}")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"imageio failed while decoding video: {video_path}") from error
    finally:
        reader.close()
    if observed_frames != frames:
        raise ValueError(f"video frame-count mismatch: {observed_frames} != {frames}")
    return {
        "width": width,
        "height": height,
        "fps": float(observed_fps),
        "decoded_frames": observed_frames,
    }


def make_video_attempt_validator(
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
) -> AttemptValidator:
    """Bind frozen protocol values to an attempt-level video validator."""

    def validate(attempt_dir: Path, completion: Mapping[str, Any]) -> None:
        videos = [row for row in completion.get("artifacts", []) if row.get("role") == "video"]
        if len(videos) != 1:
            raise ValueError("attempt must contain exactly one video artifact")
        relative = str(videos[0].get("relative_path", ""))
        video_path = _inside(attempt_dir, attempt_dir / relative)
        probe_video_protocol(
            video_path,
            width=width,
            height=height,
            fps=fps,
            frames=frames,
        )

    return validate


def build_attempt_record(
    *,
    task_id: str,
    attempt_id: int,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    run_spec_sha256: str,
    attempt_dir: Path,
    artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a completion record from files already written by generation."""

    if not task_id:
        raise ValueError("task_id is required")
    _attempt_dir_name(attempt_id)
    require_sha256(task_manifest_sha256, "task_manifest_sha256")
    require_sha256(assignment_manifest_sha256, "assignment_manifest_sha256")
    require_sha256(run_spec_sha256, "run_spec_sha256")
    if not artifacts:
        raise ValueError("at least one attempt artifact is required")
    attempt_dir = attempt_dir.resolve()
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for role, raw_path in sorted(artifacts.items()):
        if not role:
            raise ValueError("artifact role is required")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = attempt_dir / candidate
        path = _inside(attempt_dir, candidate)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(attempt_dir).as_posix()
        if relative in seen_paths:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen_paths.add(relative)
        rows.append(
            {
                "role": str(role),
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema": "hunyuan_video.stage_c_attempt_complete.v1",
        "status": "complete",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "task_manifest_sha256": task_manifest_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "run_spec_sha256": run_spec_sha256,
        "artifacts": rows,
    }
    payload["manifest_sha256"] = _self_hash(payload, "manifest_sha256")
    return payload


def validate_attempt_record(
    generation_done_path: Path,
    *,
    task_id: str,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    run_spec_sha256: str | None = None,
    required_roles: Sequence[str] = DEFAULT_REQUIRED_ARTIFACT_ROLES,
    attempt_validator: AttemptValidator | None = None,
) -> dict[str, Any]:
    """Validate identity and every declared artifact for one Stage C attempt."""

    payload = load_self_hashed_json(generation_done_path, "manifest_sha256")
    if payload.get("schema") != "hunyuan_video.stage_c_attempt_complete.v1":
        raise ValueError("unsupported Stage C attempt schema")
    if payload.get("status") != "complete" or payload.get("task_id") != task_id:
        raise ValueError("attempt status/task identity mismatch")
    attempt_id = int(payload.get("attempt_id", -1))
    expected_dir_name = _attempt_dir_name(attempt_id)
    if generation_done_path.parent.name != expected_dir_name:
        raise ValueError("attempt directory and attempt_id differ")
    expected_bindings = {
        "task_manifest_sha256": require_sha256(task_manifest_sha256, "task_manifest_sha256"),
        "assignment_manifest_sha256": require_sha256(
            assignment_manifest_sha256, "assignment_manifest_sha256"
        ),
    }
    if run_spec_sha256 is not None:
        expected_bindings["run_spec_sha256"] = require_sha256(run_spec_sha256, "run_spec_sha256")
    for name, expected in expected_bindings.items():
        if payload.get(name) != expected:
            raise ValueError(f"attempt {name} binding mismatch")
    require_sha256(payload.get("run_spec_sha256", ""), "run_spec_sha256")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("attempt artifacts must be a non-empty list")
    roles: set[str] = set()
    paths: set[str] = set()
    attempt_dir = generation_done_path.parent.resolve()
    for row in rows:
        role = str(row.get("role", ""))
        relative = str(row.get("relative_path", ""))
        if not role or not relative or role in roles or relative in paths:
            raise ValueError("attempt artifact roles and paths must be unique")
        roles.add(role)
        paths.add(relative)
        path = _inside(attempt_dir, attempt_dir / relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid artifact size: {relative}")
        if path.stat().st_size != size:
            raise ValueError(f"artifact size mismatch: {relative}")
        expected_sha = require_sha256(row.get("sha256", ""), f"artifact[{relative}].sha256")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"artifact hash mismatch: {relative}")
    missing_roles = set(required_roles) - roles
    if missing_roles:
        raise ValueError(f"attempt lacks required artifact roles: {sorted(missing_roles)}")
    if attempt_validator is not None:
        attempt_validator(attempt_dir, payload)
    return payload


def _validate_accepted_marker(
    marker: dict[str, Any],
    *,
    task_id: str,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    run_spec_sha256: str | None = None,
) -> None:
    if marker.get("schema") != "hunyuan_video.stage_c_accepted_attempt.v1":
        raise ValueError("unsupported accepted-attempt schema")
    expected = {
        "task_id": task_id,
        "task_manifest_sha256": task_manifest_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
    }
    for name, value in expected.items():
        if marker.get(name) != value:
            raise ValueError(f"accepted-attempt {name} binding mismatch")
    if run_spec_sha256 is not None and marker.get("run_spec_sha256") != run_spec_sha256:
        raise ValueError("accepted-attempt run_spec_sha256 binding mismatch")
    _attempt_dir_name(int(marker.get("accepted_attempt_id", -1)))
    require_sha256(marker.get("generation_done_sha256", ""), "generation_done_sha256")
    require_sha256(marker.get("artifact_identity_sha256", ""), "artifact_identity_sha256")


def freeze_lowest_valid_attempt(
    task_dir: Path,
    *,
    task_id: str,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    run_spec_sha256: str,
    required_roles: Sequence[str] = DEFAULT_REQUIRED_ARTIFACT_ROLES,
    attempt_validator: AttemptValidator | None = None,
) -> dict[str, Any]:
    """Freeze the lowest valid attempt; a frozen choice is never re-selected."""

    require_sha256(task_manifest_sha256, "task_manifest_sha256")
    require_sha256(assignment_manifest_sha256, "assignment_manifest_sha256")
    require_sha256(run_spec_sha256, "run_spec_sha256")
    task_dir = task_dir.resolve()
    valid: dict[int, dict[str, Any]] = {}
    for attempt_id in ATTEMPT_IDS:
        completion_path = task_dir / _attempt_dir_name(attempt_id) / "generation_done.json"
        try:
            completion = validate_attempt_record(
                completion_path,
                task_id=task_id,
                task_manifest_sha256=task_manifest_sha256,
                assignment_manifest_sha256=assignment_manifest_sha256,
                run_spec_sha256=run_spec_sha256,
                required_roles=required_roles,
                attempt_validator=attempt_validator,
            )
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        valid[attempt_id] = completion
    if len(valid) > 1:
        action_hashes = set()
        for completion in valid.values():
            actions = [row for row in completion["artifacts"] if row["role"] == "action"]
            if len(actions) != 1:
                raise ValueError("valid attempts must contain exactly one action artifact")
            action_hashes.add(actions[0]["sha256"])
        if len(action_hashes) != 1:
            raise ValueError(f"nondeterministic action artifacts for task {task_id}")

    marker_path = task_dir / "accepted_attempt.json"
    if marker_path.exists():
        marker = load_self_hashed_json(marker_path, "accepted_attempt_sha256")
        _validate_accepted_marker(
            marker,
            task_id=task_id,
            task_manifest_sha256=task_manifest_sha256,
            assignment_manifest_sha256=assignment_manifest_sha256,
            run_spec_sha256=run_spec_sha256,
        )
        attempt_id = int(marker["accepted_attempt_id"])
        completion = valid.get(attempt_id)
        if completion is None:
            raise ValueError("frozen accepted attempt is no longer valid")
        if completion["manifest_sha256"] != marker.get("generation_done_sha256"):
            raise ValueError("accepted attempt completion hash drift")
        artifact_identity = [
            {
                "role": row["role"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in completion["artifacts"]
        ]
        if hash_json(artifact_identity) != marker.get("artifact_identity_sha256"):
            raise ValueError("accepted attempt artifact identity drift")
        return marker

    if not valid:
        raise RuntimeError(f"no valid attempt for task {task_id}")
    accepted_id = min(valid)
    accepted = valid[accepted_id]
    rejected = [
        attempt_id for attempt_id in ATTEMPT_IDS if attempt_id < accepted_id and attempt_id not in valid
    ]
    artifact_identity = [
        {
            "role": row["role"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in accepted["artifacts"]
    ]
    marker = {
        "schema": "hunyuan_video.stage_c_accepted_attempt.v1",
        "task_id": task_id,
        "task_manifest_sha256": task_manifest_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "run_spec_sha256": run_spec_sha256,
        "accepted_attempt_id": accepted_id,
        "generation_done_sha256": accepted["manifest_sha256"],
        "artifact_identity_sha256": hash_json(artifact_identity),
        "rejected_lower_attempt_ids": rejected,
    }
    marker["accepted_attempt_sha256"] = _self_hash(marker, "accepted_attempt_sha256")
    write_immutable_json(marker_path, marker)
    frozen = load_self_hashed_json(marker_path, "accepted_attempt_sha256")
    _validate_accepted_marker(
        frozen,
        task_id=task_id,
        task_manifest_sha256=task_manifest_sha256,
        assignment_manifest_sha256=assignment_manifest_sha256,
        run_spec_sha256=run_spec_sha256,
    )
    return frozen


def build_generation_complete_marker(
    *,
    shard_id: str,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    expected_task_ids: Sequence[str],
    accepted_attempts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an exact-coverage shard marker from frozen accepted attempts."""

    if not shard_id:
        raise ValueError("shard_id is required")
    require_sha256(task_manifest_sha256, "task_manifest_sha256")
    require_sha256(assignment_manifest_sha256, "assignment_manifest_sha256")
    expected_ids = tuple(str(task_id) for task_id in expected_task_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected task IDs must be unique")
    rows: list[dict[str, Any]] = []
    for raw in accepted_attempts:
        marker = dict(raw)
        expected_hash = require_sha256(marker.get("accepted_attempt_sha256", ""), "accepted_attempt_sha256")
        if _self_hash(marker, "accepted_attempt_sha256") != expected_hash:
            raise ValueError("accepted-attempt self-hash mismatch")
        _validate_accepted_marker(
            marker,
            task_id=str(marker.get("task_id", "")),
            task_manifest_sha256=task_manifest_sha256,
            assignment_manifest_sha256=assignment_manifest_sha256,
        )
        rows.append(
            {
                "task_id": marker["task_id"],
                "accepted_attempt_id": int(marker["accepted_attempt_id"]),
                "accepted_attempt_sha256": marker["accepted_attempt_sha256"],
                "generation_done_sha256": require_sha256(
                    marker.get("generation_done_sha256", ""), "generation_done_sha256"
                ),
            }
        )
    observed_ids = [row["task_id"] for row in rows]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("duplicate accepted task IDs")
    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected_ids))
        raise ValueError(f"generation coverage mismatch; missing={missing}, extra={extra}")
    row_by_id = {row["task_id"]: row for row in rows}
    ordered_rows = [row_by_id[task_id] for task_id in expected_ids]
    payload: dict[str, Any] = {
        "schema": "hunyuan_video.stage_c_generation_complete.v1",
        "shard_id": shard_id,
        "task_manifest_sha256": task_manifest_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "expected_task_ids": list(expected_ids),
        "expected_task_ids_sha256": hash_json(list(expected_ids)),
        "expected_task_count": len(expected_ids),
        "accepted_task_count": len(ordered_rows),
        "accepted_attempts": ordered_rows,
    }
    payload["generation_complete_sha256"] = _self_hash(payload, "generation_complete_sha256")
    return payload


def write_generation_complete_marker(path: Path, marker: Mapping[str, Any]) -> None:
    expected = require_sha256(marker.get("generation_complete_sha256", ""), "generation_complete_sha256")
    if _self_hash(marker, "generation_complete_sha256") != expected:
        raise ValueError("generation-complete self-hash mismatch")
    write_immutable_json(path, dict(marker))


def validate_generation_complete_marker(
    path: Path,
    *,
    shard_id: str,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate one immutable worker-close marker and its exact task coverage."""

    payload = load_self_hashed_json(path, "generation_complete_sha256")
    expected_ids = tuple(str(task_id) for task_id in expected_task_ids)
    expected = {
        "schema": "hunyuan_video.stage_c_generation_complete.v1",
        "shard_id": shard_id,
        "task_manifest_sha256": require_sha256(task_manifest_sha256, "task_manifest_sha256"),
        "assignment_manifest_sha256": require_sha256(
            assignment_manifest_sha256, "assignment_manifest_sha256"
        ),
        "expected_task_ids": list(expected_ids),
        "expected_task_ids_sha256": hash_json(list(expected_ids)),
        "expected_task_count": len(expected_ids),
        "accepted_task_count": len(expected_ids),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"generation-complete {field} mismatch")
    rows = payload.get("accepted_attempts")
    if not isinstance(rows, list) or [row.get("task_id") for row in rows] != list(expected_ids):
        raise ValueError("generation-complete accepted-attempt coverage mismatch")
    for row in rows:
        if int(row.get("accepted_attempt_id", -1)) not in ATTEMPT_IDS:
            raise ValueError("generation-complete has invalid accepted attempt ID")
        require_sha256(row.get("accepted_attempt_sha256", ""), "accepted_attempt_sha256")
        require_sha256(row.get("generation_done_sha256", ""), "generation_done_sha256")
    return payload


@dataclass(frozen=True)
class DecisionRecord:
    step: int
    action: str
    reason: str
    gate_scalar: float | None = None
    accumulated: float | None = None
    threshold: float | None = None
    original_block_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
