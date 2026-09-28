from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hunyuan_video.assignment import AssignmentManifest
from hunyuan_video.backend import (
    generate,
    inference_step_count,
    load_official_sampler,
    prediction_kwargs,
    save_video,
    validate_official_source,
    verify_untouched_transformer,
)
from hunyuan_video.config import (
    RunSpec,
    TaskManifest,
    canonical_json_bytes,
    load_protocol,
    load_stage_b_tasks,
)
from hunyuan_video.records import (
    atomic_write_json,
    build_attempt_record,
    freeze_lowest_valid_attempt,
    load_self_hashed_json,
    make_video_attempt_validator,
    sha256_file,
    validate_attempt_record,
    write_immutable_json,
)


ROOT = Path(__file__).resolve().parents[1]
_STAGE_B_COMPLETION_SCHEMA = "hunyuan_video.stage_b_generation_done.v1"
_C1_REQUIRED_ARTIFACT_ROLES = ("video", "action", "generation_metadata")


@dataclass(frozen=True)
class _AssignedTask:
    run_spec: RunSpec
    formal_row_id: str | None = None
    formal_row_sha256: str | None = None


@dataclass(frozen=True)
class _FormalExecution:
    task_manifest: TaskManifest
    assignment_manifest: AssignmentManifest
    tasks: tuple[_AssignedTask, ...]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _require_compute_node() -> None:
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("HunyuanVideo generation must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo generation requires CUDA")


def _repo_identity() -> dict[str, Any]:
    return {"git_sha": None, "status_clean": None}


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    array = value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "mean": float(value.float().mean().item()),
        "std": float(value.float().std().item()),
        "min": float(value.float().min().item()),
        "max": float(value.float().max().item()),
    }


def _parity(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = candidate.detach().float().cpu() - reference.float()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms": float(delta.square().mean().sqrt().item()),
    }


def _fingerprint(tensor: torch.Tensor, count: int = 65536) -> torch.Tensor:
    flat = tensor.detach().float().cpu().reshape(-1)
    if flat.numel() <= count:
        return flat.clone()
    indices = torch.linspace(0, flat.numel() - 1, count, dtype=torch.float64).round().long()
    return flat[indices].clone()


def _atomic_save_npy(path: Path, tensor: torch.Tensor) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, tensor.numpy(), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _publish_immutable_temp(tmp: Path, path: Path) -> None:
    try:
        os.link(tmp, path)
    except FileExistsError as error:
        raise FileExistsError(f"immutable artifact already exists: {path}") from error
    finally:
        if tmp.exists():
            tmp.unlink()


def _immutable_save_npy(path: Path, tensor: torch.Tensor) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.immutable.tmp")
    with tmp.open("xb") as handle:
        np.save(handle, tensor.numpy(), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_immutable_temp(tmp, path)


def _immutable_torch_save(path: Path, tensor: torch.Tensor) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        torch.save(tensor.detach().contiguous().cpu(), handle)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_immutable_temp(tmp, path)


def _immutable_save_video(api: dict[str, Any], sample: torch.Tensor, path: Path, fps: int) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.immutable{path.suffix}")
    if tmp.exists():
        raise FileExistsError(f"stale immutable video temporary file: {tmp}")
    try:
        save_video(api, sample, tmp, fps)
        if not tmp.is_file():
            raise RuntimeError(f"video backend did not create expected file: {tmp}")
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_immutable_temp(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _claim_formal_attempt(
    task_dir: Path,
    *,
    task: _AssignedTask,
    task_manifest_sha256: str,
    assignment_manifest_sha256: str,
    attempt_index: int,
    worker_id: int,
) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    existing = list(task_dir.iterdir())
    if existing:
        raise RuntimeError(
            f"incomplete immutable attempt already exists for {task.run_spec.task_id}; "
            "use the next attempt index"
        )
    claim_path = task_dir / "generation_claim.json"
    payload = {
        "schema": "hunyuan_video.stage_c_generation_claim.v1",
        "task_id": task.run_spec.task_id,
        "run_spec_sha256": task.run_spec.sha256,
        "formal_row_id": task.formal_row_id,
        "task_manifest_sha256": task_manifest_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "attempt_index": attempt_index,
        "worker_id": worker_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    data = canonical_json_bytes(_self_hashed(payload)) + b"\n"
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"attempt is already claimed: {task.run_spec.task_id}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def _stage_b_task_dir(output_root: Path, task: RunSpec) -> Path:
    return output_root / task.phase / task.protocol_id / task.mode / task.task_id / "attempt_000"


def _formal_task_dir(output_root: Path, task: _AssignedTask, attempt_index: int) -> Path:
    assert task.formal_row_id is not None
    return (
        output_root
        / task.run_spec.phase
        / task.formal_row_id
        / task.run_spec.task_id
        / f"attempt_{attempt_index:03d}"
    )


def _task_dir(output_root: Path, task: RunSpec) -> Path:
    """Stage B compatibility helper retained for existing callers/tests."""
    return _stage_b_task_dir(output_root, task)


def _shard_key(task: RunSpec) -> tuple[str, ...]:
    """Keep Stage B repetitions that benefit from model reuse on one worker."""
    if task.phase == "identity":
        return (task.prompt_id,)
    if task.phase == "action":
        return (task.prompt_id, task.mode)
    if task.phase == "pilot":
        return (task.prompt_id, task.protocol_id, task.mode)
    raise ValueError(f"unsupported Stage B phase: {task.phase}")


def _tasks_for_shard(tasks: list[RunSpec], shard_index: int, shard_count: int) -> list[RunSpec]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    keys = sorted({_shard_key(task) for task in tasks})
    assigned = {key for index, key in enumerate(keys) if index % shard_count == shard_index}
    return [task for task in tasks if _shard_key(task) in assigned]


def bounded_start_delay(
    *,
    node_id: int,
    local_worker_id: int,
    launch_nodes: int,
    workers_per_node: int,
    node_batch_size: int,
    batch_delay_seconds: int,
    local_delay_seconds: int,
    max_stagger_seconds: int,
) -> int:
    """Return a frozen node-batch launch delay and prove its global bound."""

    values = {
        "node_id": node_id,
        "local_worker_id": local_worker_id,
        "launch_nodes": launch_nodes,
        "workers_per_node": workers_per_node,
        "node_batch_size": node_batch_size,
        "batch_delay_seconds": batch_delay_seconds,
        "local_delay_seconds": local_delay_seconds,
        "max_stagger_seconds": max_stagger_seconds,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
        raise TypeError("stagger parameters must be integers")
    if launch_nodes < 1 or workers_per_node < 1 or node_batch_size < 1:
        raise ValueError("launch topology and node batch size must be positive")
    if batch_delay_seconds < 0 or local_delay_seconds < 0 or max_stagger_seconds < 0:
        raise ValueError("stagger delays must be non-negative")
    if not 0 <= node_id < launch_nodes:
        raise ValueError(f"node_id out of range: {node_id}/{launch_nodes}")
    if not 0 <= local_worker_id < workers_per_node:
        raise ValueError(f"local_worker_id out of range: {local_worker_id}/{workers_per_node}")
    last_batch = (launch_nodes - 1) // node_batch_size
    maximum = last_batch * batch_delay_seconds + (workers_per_node - 1) * local_delay_seconds
    if maximum > max_stagger_seconds:
        raise ValueError(f"stagger configuration exceeds bound: {maximum} > {max_stagger_seconds} seconds")
    return (node_id // node_batch_size) * batch_delay_seconds + local_worker_id * local_delay_seconds


def _load_formal_execution(
    task_manifest_path: Path,
    assignment_manifest_path: Path,
    worker_id: int | None,
) -> _FormalExecution:
    task_manifest = TaskManifest.load(task_manifest_path)
    assignment = AssignmentManifest.load(assignment_manifest_path)
    if assignment.task_manifest_sha256 != task_manifest.sha256:
        raise ValueError("assignment/task manifest hash mismatch")
    task_by_id = task_manifest.task_by_id()
    row_by_id = {row.row_id: row for row in task_manifest.formal_rows}
    if worker_id is None:
        task_ids = tuple(task_id for worker in assignment.workers for task_id in worker.ordered_task_ids)
    else:
        task_ids = assignment.tasks_for_worker(worker_id)
    assigned: list[_AssignedTask] = []
    for task_id in task_ids:
        formal_task = task_by_id.get(task_id)
        if formal_task is None:
            raise ValueError(f"assignment references unknown task: {task_id}")
        row = row_by_id[formal_task.formal_row_id]
        assigned.append(
            _AssignedTask(
                run_spec=formal_task.run_spec,
                formal_row_id=row.row_id,
                formal_row_sha256=row.sha256,
            )
        )
    return _FormalExecution(task_manifest, assignment, tuple(assigned))


def _direct_generate(sampler: Any, protocol: Any, task: RunSpec) -> tuple[dict[str, Any], float, int]:
    verify_untouched_transformer(sampler.pipeline.transformer)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = sampler.predict(**prediction_kwargs(protocol, task))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    verify_untouched_transformer(sampler.pipeline.transformer)
    return output, elapsed, peak


def _self_hashed(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _read_self_hashed(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read completion record {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"completion record is not a JSON object: {path}")
    declared = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    try:
        actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"completion record is not canonical JSON: {path}") from error
    if declared != actual:
        raise RuntimeError(f"completion self-hash mismatch: {path}")
    return payload


def _validate_artifact(task_dir: Path, descriptor: Any, label: str) -> None:
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"completion lacks {label} artifact descriptor")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"invalid {label} artifact path")
    path = task_dir / relative
    try:
        path.resolve().relative_to(task_dir.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} artifact escapes attempt directory") from error
    if not path.is_file():
        raise RuntimeError(f"missing {label} artifact: {path}")
    if "bytes" in descriptor and path.stat().st_size != descriptor["bytes"]:
        raise RuntimeError(f"{label} artifact size mismatch: {path}")
    if sha256_file(path) != descriptor.get("sha256"):
        raise RuntimeError(f"{label} artifact hash mismatch: {path}")


def _validate_completion(
    done_path: Path,
    task: _AssignedTask,
    repo: dict[str, Any],
    official: dict[str, Any],
    protocol: Any,
    *,
    task_manifest_sha256: str | None = None,
    assignment_manifest_sha256: str | None = None,
    attempt_index: int = 0,
) -> dict[str, Any]:
    run = task.run_spec
    if task.formal_row_id is not None:
        try:
            payload = validate_attempt_record(
                done_path,
                task_id=run.task_id,
                task_manifest_sha256=str(task_manifest_sha256),
                assignment_manifest_sha256=str(assignment_manifest_sha256),
                run_spec_sha256=run.sha256,
                required_roles=_C1_REQUIRED_ARTIFACT_ROLES,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid formal completion for {run.task_id}: {error}") from error
        if int(payload["attempt_id"]) != attempt_index:
            raise RuntimeError(f"completion attempt index mismatch for {run.task_id}")
        artifacts = {row["role"]: row for row in payload["artifacts"]}
        metadata_row = artifacts.get("generation_metadata")
        if metadata_row is None:
            raise RuntimeError(f"completion lacks generation metadata for {run.task_id}")
        metadata = load_self_hashed_json(done_path.parent / metadata_row["relative_path"], "manifest_sha256")
        expected_metadata = {
            "schema": "hunyuan_video.stage_c_generation_metadata.v1",
            "task_id": run.task_id,
            "run_spec_sha256": run.sha256,
            "formal_row_id": task.formal_row_id,
            "formal_row_sha256": task.formal_row_sha256,
            "task_manifest_sha256": task_manifest_sha256,
            "assignment_manifest_sha256": assignment_manifest_sha256,
            "attempt_index": attempt_index,
            "repo": repo,
            "official_source": official,
            "protocol_manifest_sha256": protocol.manifest_sha256,
            "protocol_sha256": protocol.protocol_sha256,
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise RuntimeError(f"generation metadata {field} mismatch for {run.task_id}")
        action_row = artifacts["action"]
        action = load_self_hashed_json(done_path.parent / action_row["relative_path"], "manifest_sha256")
        if action.get("task_id") != run.task_id or action.get("run_spec_sha256") != run.sha256:
            raise RuntimeError(f"action identity mismatch for {run.task_id}")
        return payload

    payload = _read_self_hashed(done_path)
    exact = {
        "schema": _STAGE_B_COMPLETION_SCHEMA,
        "task_id": run.task_id,
        "run_spec_sha256": run.sha256,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "protocol_sha256": protocol.protocol_sha256,
    }
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"completion {field} mismatch for {run.task_id}")
    task_dir = done_path.parent
    _validate_artifact(task_dir, payload.get("sample_fingerprint"), "sample fingerprint")
    _validate_artifact(task_dir, payload.get("video"), "video")
    return payload


def _natural_final_latent(output: dict[str, Any]) -> tuple[str, torch.Tensor] | None:
    for key in ("final_latent", "final_latents", "latents"):
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            return key, value
    return None


def _validate_protocol_binding(task: _AssignedTask, execution: _FormalExecution | None) -> Any:
    protocol = load_protocol(task.run_spec.protocol_id)
    if execution is not None:
        expected = execution.task_manifest.protocol_manifest_sha256s[protocol.protocol_id]
        if protocol.manifest_sha256 != expected:
            raise RuntimeError(
                f"protocol manifest drift for {protocol.protocol_id}: "
                f"{protocol.manifest_sha256} != {expected}"
            )
    return protocol


def _protocol_attempt_validator(protocol: Any) -> Any:
    return make_video_attempt_validator(
        width=protocol.width,
        height=protocol.height,
        fps=protocol.fps,
        frames=protocol.frames,
    )


def _action_records(
    *,
    adapter: Any | None,
    transformer: Any | None,
    protocol: Any,
    task: RunSpec,
    formal: bool,
) -> tuple[int, list[dict[str, Any]]]:
    executed_steps = inference_step_count(protocol, task)
    if formal and adapter is None:
        if transformer is None:
            raise RuntimeError("formal original action records require the official transformer")
        block_calls = len(transformer.double_blocks) + len(transformer.single_blocks)
        records = [
            {
                "step": step,
                "action": "full",
                "reason": "official_full",
                "gate_scalar": None,
                "accumulated": None,
                "threshold": None,
                "original_block_calls": block_calls,
            }
            for step in range(executed_steps)
        ]
    else:
        records = [] if adapter is None else [row.as_dict() for row in adapter.decisions]
    if adapter is not None and len(records) != executed_steps:
        raise RuntimeError(
            f"adapter action count differs from executed steps: {len(records)} != {executed_steps}"
        )
    return executed_steps, records


def _has_valid_prior_result(
    *,
    task_dir: Path,
    task: _AssignedTask,
    repo: dict[str, Any],
    official: dict[str, Any],
    protocol: Any,
    execution: _FormalExecution,
    attempt_index: int,
) -> bool:
    if attempt_index != 1:
        return False
    task_root = task_dir.parent
    validator = _protocol_attempt_validator(protocol)
    marker_path = task_root / "accepted_attempt.json"
    if marker_path.exists():
        freeze_lowest_valid_attempt(
            task_root,
            task_id=task.run_spec.task_id,
            task_manifest_sha256=execution.task_manifest.sha256,
            assignment_manifest_sha256=execution.assignment_manifest.sha256,
            run_spec_sha256=task.run_spec.sha256,
            required_roles=_C1_REQUIRED_ARTIFACT_ROLES,
            attempt_validator=validator,
        )
        return True
    prior_done = task_root / "attempt_000" / "generation_done.json"
    try:
        validate_attempt_record(
            prior_done,
            task_id=task.run_spec.task_id,
            task_manifest_sha256=execution.task_manifest.sha256,
            assignment_manifest_sha256=execution.assignment_manifest.sha256,
            run_spec_sha256=task.run_spec.sha256,
            required_roles=_C1_REQUIRED_ARTIFACT_ROLES,
            attempt_validator=validator,
        )
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    _validate_completion(
        prior_done,
        task,
        repo,
        official,
        protocol,
        task_manifest_sha256=execution.task_manifest.sha256,
        assignment_manifest_sha256=execution.assignment_manifest.sha256,
        attempt_index=0,
    )
    freeze_lowest_valid_attempt(
        task_root,
        task_id=task.run_spec.task_id,
        task_manifest_sha256=execution.task_manifest.sha256,
        assignment_manifest_sha256=execution.assignment_manifest.sha256,
        run_spec_sha256=task.run_spec.sha256,
        required_roles=_C1_REQUIRED_ARTIFACT_ROLES,
        attempt_validator=validator,
    )
    return True


def _validate_repo_binding(repo: dict[str, Any], execution: _FormalExecution | None) -> None:
    return


def _preflight(
    args: argparse.Namespace,
    repo: dict[str, Any],
    official: dict[str, Any],
    execution: _FormalExecution,
) -> int:
    _validate_repo_binding(repo, execution)
    if args.expected_worker_count is not None:
        observed = len(execution.assignment_manifest.workers)
        if observed != args.expected_worker_count:
            raise RuntimeError(
                f"assignment worker count mismatch: {observed} != {args.expected_worker_count}"
            )
    if (
        args.expected_assignment_nodes is not None
        and execution.assignment_manifest.node_count != args.expected_assignment_nodes
    ):
        raise RuntimeError(
            "assignment node count mismatch: "
            f"{execution.assignment_manifest.node_count} != {args.expected_assignment_nodes}"
        )
    if (
        args.expected_assignment_workers_per_node is not None
        and execution.assignment_manifest.workers_per_node != args.expected_assignment_workers_per_node
    ):
        raise RuntimeError(
            "assignment workers-per-node mismatch: "
            f"{execution.assignment_manifest.workers_per_node} "
            f"!= {args.expected_assignment_workers_per_node}"
        )
    protocols: dict[str, Any] = {}
    checked = 0
    for task in execution.tasks:
        protocol = protocols.setdefault(
            task.run_spec.protocol_id,
            _validate_protocol_binding(task, execution),
        )
        task_dir = _formal_task_dir(args.output_root, task, args.attempt_index)
        done_path = task_dir / "generation_done.json"
        if done_path.exists():
            if not done_path.is_file():
                raise RuntimeError(f"completion path is not a file: {done_path}")
            _validate_completion(
                done_path,
                task,
                repo,
                official,
                protocol,
                task_manifest_sha256=execution.task_manifest.sha256,
                assignment_manifest_sha256=execution.assignment_manifest.sha256,
                attempt_index=args.attempt_index,
            )
            checked += 1
    print(
        json.dumps(
            {
                "preflight": "pass",
                "tasks": len(execution.tasks),
                "existing_completions_checked": checked,
                "task_manifest_sha256": execution.task_manifest.sha256,
                "assignment_manifest_sha256": execution.assignment_manifest.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def run(args: argparse.Namespace) -> int:
    _require_compute_node()
    repo = _repo_identity()
    official = validate_official_source()
    formal = args.assignment_manifest is not None
    execution: _FormalExecution | None = None
    if formal:
        execution = _load_formal_execution(
            args.task_manifest,
            args.assignment_manifest,
            None if args.preflight_only else args.worker_id,
        )
        _validate_repo_binding(repo, execution)
        if args.preflight_only:
            return _preflight(args, repo, official, execution)
        tasks = list(execution.tasks)
    else:
        stage_b_tasks = [task for task in load_stage_b_tasks(args.task_manifest) if task.phase == args.phase]
        tasks = [
            _AssignedTask(task)
            for task in _tasks_for_shard(stage_b_tasks, args.shard_index, args.shard_count)
        ]
    if not tasks:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "worker": args.worker_id if formal else args.shard_index,
                    "tasks": 0,
                }
            )
        )
        return 0

    if formal and args.launch_nodes is not None:
        delay = bounded_start_delay(
            node_id=args.node_id,
            local_worker_id=args.local_worker_id,
            launch_nodes=args.launch_nodes,
            workers_per_node=args.workers_per_node,
            node_batch_size=args.node_batch_size,
            batch_delay_seconds=args.batch_delay_seconds,
            local_delay_seconds=args.local_delay_seconds,
            max_stagger_seconds=args.max_stagger_seconds,
        )
        if delay:
            print(
                json.dumps(
                    {
                        "worker_id": args.worker_id,
                        "node_id": args.node_id,
                        "local_worker_id": args.local_worker_id,
                        "startup_delay_seconds": delay,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)

    sampler = None
    api = None
    loaded_protocol = None
    load_evidence: dict[str, Any] | None = None
    identity_references: dict[str, torch.Tensor] = {}
    completed = 0
    generated = 0
    skipped_prior = 0
    for assigned_task in tasks:
        task = assigned_task.run_spec
        protocol = _validate_protocol_binding(assigned_task, execution)
        task_dir = (
            _formal_task_dir(args.output_root, assigned_task, args.attempt_index)
            if formal
            else _stage_b_task_dir(args.output_root, task)
        )
        done_path = task_dir / "generation_done.json"
        if done_path.exists():
            if not done_path.is_file():
                raise RuntimeError(f"completion path is not a file: {done_path}")
            _validate_completion(
                done_path,
                assigned_task,
                repo,
                official,
                protocol,
                task_manifest_sha256=execution.task_manifest.sha256 if execution else None,
                assignment_manifest_sha256=(execution.assignment_manifest.sha256 if execution else None),
                attempt_index=args.attempt_index,
            )
            if formal:
                assert execution is not None
                freeze_lowest_valid_attempt(
                    task_dir.parent,
                    task_id=task.task_id,
                    task_manifest_sha256=execution.task_manifest.sha256,
                    assignment_manifest_sha256=execution.assignment_manifest.sha256,
                    run_spec_sha256=task.sha256,
                    required_roles=_C1_REQUIRED_ARTIFACT_ROLES,
                    attempt_validator=_protocol_attempt_validator(protocol),
                )
            if task.phase == "identity" and task.prompt_id not in identity_references:
                identity_references[task.prompt_id] = torch.from_numpy(
                    np.load(task_dir / "sample_fingerprint.npy", allow_pickle=False)
                ).float()
            completed += 1
            continue
        if formal and _has_valid_prior_result(
            task_dir=task_dir,
            task=assigned_task,
            repo=repo,
            official=official,
            protocol=protocol,
            execution=execution,
            attempt_index=args.attempt_index,
        ):
            completed += 1
            skipped_prior += 1
            continue
        if formal and task_dir.exists() and any(task_dir.iterdir()):
            raise RuntimeError(
                f"incomplete immutable attempt already exists for {task.task_id}; "
                "use the next attempt index"
            )
        if loaded_protocol != task.protocol_id:
            del sampler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sampler, api, load_evidence = load_official_sampler(args.model_base, protocol)
            loaded_protocol = task.protocol_id
        assert sampler is not None and api is not None and load_evidence is not None
        claim_path = None
        if formal:
            claim_path = _claim_formal_attempt(
                task_dir,
                task=assigned_task,
                task_manifest_sha256=execution.task_manifest.sha256,
                assignment_manifest_sha256=execution.assignment_manifest.sha256,
                attempt_index=args.attempt_index,
                worker_id=args.worker_id,
            )
        else:
            task_dir.mkdir(parents=True, exist_ok=True)
        run_record: dict[str, Any] = {
            **task.as_dict(),
            "run_spec_sha256": task.sha256,
        }
        if formal:
            run_record.update(
                {
                    "formal_row_id": assigned_task.formal_row_id,
                    "formal_row_sha256": assigned_task.formal_row_sha256,
                    "task_manifest_sha256": execution.task_manifest.sha256,
                    "assignment_manifest_sha256": execution.assignment_manifest.sha256,
                    "attempt_index": args.attempt_index,
                }
            )
        if formal:
            write_immutable_json(task_dir / "run_spec.json", run_record)
        else:
            atomic_write_json(task_dir / "run_spec.json", run_record)
        entrypoint = task.method_config.get("entrypoint", "backend")
        if task.mode == "original" and entrypoint == "official-direct":
            output, generation_seconds, peak_vram = _direct_generate(sampler, protocol, task)
            adapter = None
        else:
            output, adapter, generation_seconds, peak_vram = generate(sampler, protocol, task)
        samples = output["samples"]
        if len(samples) != 1:
            raise RuntimeError(f"expected one sample, got {len(samples)}")
        sample = samples[0]
        fingerprint = _fingerprint(sample)
        fingerprint_path = task_dir / "sample_fingerprint.npy"
        if formal:
            _immutable_save_npy(fingerprint_path, fingerprint)
        else:
            _atomic_save_npy(fingerprint_path, fingerprint)
        video_path = task_dir / "video.mp4"
        if formal:
            _immutable_save_video(api, sample, video_path, protocol.fps)
        else:
            save_video(api, sample, video_path, protocol.fps)
        parity = None
        if task.phase == "identity":
            if task.prompt_id not in identity_references:
                identity_references[task.prompt_id] = fingerprint
                parity = {"max_abs": 0.0, "mean_abs": 0.0, "rms": 0.0}
            else:
                parity = _parity(fingerprint, identity_references[task.prompt_id])
        executed_steps, decisions = _action_records(
            adapter=adapter,
            transformer=getattr(getattr(sampler, "pipeline", None), "transformer", None),
            protocol=protocol,
            task=task,
            formal=formal,
        )
        full_count = sum(row["action"] == "full" for row in decisions)
        cache_count = sum(row["action"] == "cache" for row in decisions)
        decision_summary = {
            "num_steps": executed_steps,
            "full_count": full_count,
            "cache_count": cache_count,
            "cache_ratio": cache_count / executed_steps if decisions else 0.0,
            "records": decisions,
        }
        action_path = None
        if formal:
            action_path = task_dir / "actions.json"
            action_record = _self_hashed(
                {
                    "schema": "hunyuan_video.stage_c_actions.v1",
                    "task_id": task.task_id,
                    "run_spec_sha256": task.sha256,
                    "formal_row_id": assigned_task.formal_row_id,
                    "protocol_sha256": protocol.protocol_sha256,
                    "num_steps": executed_steps,
                    "full_count": full_count,
                    "cache_count": cache_count,
                    "cache_ratio": cache_count / executed_steps if decisions else 0.0,
                    "records": decisions,
                }
            )
            write_immutable_json(action_path, action_record)
        latent_path = None
        latent_metadata = None
        natural_latent = _natural_final_latent(output)
        if formal and natural_latent is not None:
            latent_key, latent = natural_latent
            latent_path = task_dir / "final_latent.pt"
            _immutable_torch_save(latent_path, latent)
            latent_metadata = {
                "output_key": latent_key,
                "tensor": _tensor_digest(latent),
            }
        generation_metadata: dict[str, Any] = {
            "schema": "hunyuan_video.stage_c_generation_metadata.v1",
            "task_id": task.task_id,
            "run_spec_sha256": task.sha256,
            "repo": repo,
            "official_source": official,
            "protocol_manifest_sha256": protocol.manifest_sha256,
            "protocol_sha256": protocol.protocol_sha256,
            "entrypoint": entrypoint,
            "executed_inference_steps": executed_steps,
            "load_seconds": load_evidence["load_seconds"],
            "generation_seconds": generation_seconds,
            "peak_vram_bytes": peak_vram,
            "sample": _tensor_digest(sample),
            "sample_fingerprint": {
                "path": fingerprint_path.name,
                "count": int(fingerprint.numel()),
                "bytes": fingerprint_path.stat().st_size,
                "sha256": sha256_file(fingerprint_path),
            },
            "video": {
                "path": video_path.name,
                "bytes": video_path.stat().st_size,
                "sha256": sha256_file(video_path),
            },
            "parity_to_first_repeat": parity,
            "decision_summary": decision_summary,
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "shard_index": args.shard_index if not formal else None,
                "shard_count": args.shard_count if not formal else None,
                "worker_id": args.worker_id if formal else None,
            },
        }
        if formal:
            generation_metadata.update(
                {
                    "formal_row_id": assigned_task.formal_row_id,
                    "formal_row_sha256": assigned_task.formal_row_sha256,
                    "task_manifest_sha256": execution.task_manifest.sha256,
                    "assignment_manifest_sha256": execution.assignment_manifest.sha256,
                    "attempt_index": args.attempt_index,
                    "final_latent": latent_metadata,
                }
            )
            metadata_path = task_dir / "generation_metadata.json"
            write_immutable_json(metadata_path, _self_hashed(generation_metadata))
            assert action_path is not None and claim_path is not None
            artifacts = {
                "video": video_path,
                "action": action_path,
                "sample_fingerprint": fingerprint_path,
                "generation_metadata": metadata_path,
                "generation_claim": claim_path,
            }
            if latent_path is not None:
                artifacts["final_latent"] = latent_path
            completion = build_attempt_record(
                task_id=task.task_id,
                attempt_id=args.attempt_index,
                task_manifest_sha256=execution.task_manifest.sha256,
                assignment_manifest_sha256=execution.assignment_manifest.sha256,
                run_spec_sha256=task.sha256,
                attempt_dir=task_dir,
                artifacts=artifacts,
            )
            write_immutable_json(done_path, completion)
            freeze_lowest_valid_attempt(
                task_dir.parent,
                task_id=task.task_id,
                task_manifest_sha256=execution.task_manifest.sha256,
                assignment_manifest_sha256=execution.assignment_manifest.sha256,
                run_spec_sha256=task.sha256,
                required_roles=_C1_REQUIRED_ARTIFACT_ROLES,
                attempt_validator=_protocol_attempt_validator(protocol),
            )
        else:
            generation_metadata["schema"] = _STAGE_B_COMPLETION_SCHEMA
            atomic_write_json(done_path, _self_hashed(generation_metadata))
        completed += 1
        generated += 1
        del adapter, output, samples, sample, fingerprint
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "worker": args.worker_id if formal else args.shard_index,
                "completed": completed,
                "generated": generated,
                "skipped_prior": skipped_prior,
            }
        )
    )
    return 0


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    formal = args.assignment_manifest is not None
    if formal:
        if args.attempt_index not in (0, 1):
            parser.error("formal --attempt-index must be 0 or 1")
        if args.preflight_only:
            if args.worker_id is not None:
                parser.error("--preflight-only scans all workers; omit --worker-id")
        elif args.worker_id is None:
            parser.error("formal execution requires --worker-id")
        if args.phase is not None or args.shard_index is not None or args.shard_count is not None:
            parser.error("formal execution does not accept Stage B phase/shard arguments")
        stagger_values = (
            args.launch_nodes,
            args.workers_per_node,
            args.node_id,
            args.local_worker_id,
            args.node_batch_size,
            args.batch_delay_seconds,
            args.local_delay_seconds,
            args.max_stagger_seconds,
        )
        if any(value is not None for value in stagger_values) and not all(
            value is not None for value in stagger_values
        ):
            parser.error("formal stagger arguments must be supplied together")
        if args.preflight_only and any(value is not None for value in stagger_values):
            parser.error("preflight does not accept worker stagger arguments")
    else:
        if args.preflight_only:
            parser.error("--preflight-only requires --assignment-manifest")
        if args.worker_id is not None or args.attempt_index != 0:
            parser.error("worker/attempt arguments require --assignment-manifest")
        if args.phase is None or args.shard_index is None or args.shard_count is None:
            parser.error("Stage B execution requires --phase, --shard-index, and --shard-count")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("identity", "action", "pilot"))
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path)
    parser.add_argument("--model-base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--expected-worker-count", type=int)
    parser.add_argument("--expected-assignment-nodes", type=int)
    parser.add_argument("--expected-assignment-workers-per-node", type=int)
    parser.add_argument("--launch-nodes", type=int)
    parser.add_argument("--workers-per-node", type=int)
    parser.add_argument("--node-id", type=int)
    parser.add_argument("--local-worker-id", type=int)
    parser.add_argument("--node-batch-size", type=int)
    parser.add_argument("--batch-delay-seconds", type=int)
    parser.add_argument("--local-delay-seconds", type=int)
    parser.add_argument("--max-stagger-seconds", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
