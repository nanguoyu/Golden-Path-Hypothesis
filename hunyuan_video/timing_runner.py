from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import torch

from hunyuan_video.backend import (
    generate,
    inference_step_count,
    load_official_sampler,
    validate_official_source,
)
from hunyuan_video.config import TaskManifest, hash_json, load_protocol
from hunyuan_video.records import load_self_hashed_json, write_immutable_json


ROOT = Path(__file__).resolve().parents[1]


def _load_assignment(path: Path, manifest: TaskManifest) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if hash_json(body) != expected:
        raise ValueError("T16 assignment self-hash mismatch")
    if (
        payload.get("schema") != "hunyuan_video.stage_c_t16_assignment.v1"
        or payload.get("task_manifest_sha256") != manifest.sha256
        or payload.get("node_count") != 8
        or len(payload.get("workers", [])) != 8
    ):
        raise ValueError("T16 assignment identity/topology mismatch")
    task_ids = [
        task_id
        for worker in payload["workers"]
        for task_id in worker["ordered_task_ids"]
    ]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("T16 assignment contains duplicate task IDs")
    task_by_id = manifest.task_by_id()
    if any(task_id not in task_by_id for task_id in task_ids):
        raise ValueError("T16 assignment references an unknown task")
    if any(task_by_id[task_id].run_spec.repeat != payload["repeat"] for task_id in task_ids):
        raise ValueError("T16 assignment mixes repeat waves")
    return payload


def _repo_identity() -> dict[str, Any]:
    return {"git_sha": None, "status_clean": None}


def _require_compute_node() -> None:
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("T16 timing must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("T16 timing requires CUDA")


def _record_path(output_root: Path, lane: str, repeat: int, row_id: str, task_id: str) -> Path:
    return output_root / lane / f"repeat_{repeat}" / row_id / f"{task_id}.json"


def _validate_existing(
    path: Path,
    *,
    task_id: str,
    run_sha256: str,
    task_manifest_sha256: str,
    assignment_sha256: str,
) -> None:
    payload = load_self_hashed_json(path, "manifest_sha256")
    expected = {
        "schema": "hunyuan_video.stage_c_t16_timing.v1",
        "task_id": task_id,
        "run_spec_sha256": run_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "assignment_manifest_sha256": assignment_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"T16 existing record identity mismatch: {path}")


def _run_generation(sampler: Any, protocol: Any, run: Any) -> tuple[dict[str, Any], float, int]:
    output, adapter, seconds, peak_vram = generate(sampler, protocol, run)
    steps = inference_step_count(protocol, run)
    if adapter is None:
        full_count, cache_count = steps, 0
    else:
        records = adapter.decisions
        if len(records) != steps:
            raise RuntimeError("T16 action count differs from executed denoise steps")
        full_count = sum(record.action == "full" for record in records)
        cache_count = sum(record.action == "cache" for record in records)
    del output, adapter
    return {
        "cache_count": cache_count,
        "cache_ratio": cache_count / steps,
        "full_count": full_count,
        "num_steps": steps,
    }, seconds, peak_vram


def run(args: argparse.Namespace) -> int:
    _require_compute_node()
    manifest = TaskManifest.load(args.task_manifest)
    assignment = _load_assignment(args.assignment_manifest, manifest)
    if assignment["lane"] != manifest.metadata.get("lane"):
        raise ValueError("T16 assignment lane differs from task manifest")
    if assignment["repeat"] != args.repeat:
        raise ValueError("T16 CLI repeat differs from assignment")
    if not 0 <= args.worker_id < 8:
        raise ValueError("T16 worker ID must be in [0, 7]")
    repo = _repo_identity()
    official = validate_official_source()
    task_by_id = manifest.task_by_id()
    worker = assignment["workers"][args.worker_id]
    if worker["worker_id"] != args.worker_id:
        raise ValueError("T16 worker ordering drift")
    tasks = [task_by_id[task_id] for task_id in worker["ordered_task_ids"]]
    protocol_ids = {task.run_spec.protocol_id for task in tasks}
    if len(protocol_ids) != 1:
        raise ValueError("one T16 wave worker must use exactly one generation protocol")
    protocol = load_protocol(next(iter(protocol_ids)))
    if manifest.protocol_manifest_sha256s[protocol.protocol_id] != protocol.manifest_sha256:
        raise ValueError("T16 protocol manifest drift")
    sampler, _api, load_evidence = load_official_sampler(args.model_base, protocol)

    first_by_row: dict[str, Any] = {}
    for task in tasks:
        first_by_row.setdefault(task.formal_row_id, task)
    if set(first_by_row) != set(assignment["warmup_row_ids"]):
        raise ValueError("each T16 worker must cover every warmup row")
    for row_id in assignment["warmup_row_ids"]:
        _run_generation(sampler, protocol, first_by_row[row_id].run_spec)

    completed = 0
    for task in tasks:
        run_spec = task.run_spec
        output_path = _record_path(
            args.output_root,
            assignment["lane"],
            args.repeat,
            task.formal_row_id,
            run_spec.task_id,
        )
        if output_path.exists():
            _validate_existing(
                output_path,
                task_id=run_spec.task_id,
                run_sha256=run_spec.sha256,
                task_manifest_sha256=manifest.sha256,
                assignment_sha256=assignment["manifest_sha256"],
            )
            completed += 1
            continue
        action, generation_seconds, peak_vram = _run_generation(
            sampler, protocol, run_spec
        )
        record = {
            "schema": "hunyuan_video.stage_c_t16_timing.v1",
            "action": action,
            "assignment_manifest_sha256": assignment["manifest_sha256"],
            "formal_row_id": task.formal_row_id,
            "generation_seconds": generation_seconds,
            "hostname": socket.gethostname(),
            "lane": assignment["lane"],
            "load_evidence": load_evidence,
            "peak_vram_bytes": peak_vram,
            "protocol_manifest_sha256": protocol.manifest_sha256,
            "protocol_sha256": protocol.protocol_sha256,
            "repeat": args.repeat,
            "repo": repo,
            "run_spec_sha256": run_spec.sha256,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source": official,
            "task_id": run_spec.task_id,
            "task_manifest_sha256": manifest.sha256,
            "worker_id": args.worker_id,
        }
        record["manifest_sha256"] = hash_json(record)
        write_immutable_json(output_path, record)
        completed += 1
    print(
        json.dumps(
            {
                "completed": completed,
                "lane": assignment["lane"],
                "repeat": args.repeat,
                "worker_id": args.worker_id,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--model-base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
