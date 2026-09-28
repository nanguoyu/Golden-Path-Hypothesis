from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import TaskManifest, canonical_json_bytes, hash_json, require_sha256


LPT_ALGORITHM = "deterministic-lpt-int-v1"


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    ordered_task_ids: tuple[str, ...]
    estimated_cost: int

    def __post_init__(self) -> None:
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        if len(set(self.ordered_task_ids)) != len(self.ordered_task_ids):
            raise ValueError(f"worker {self.worker_id} has duplicate task IDs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "ordered_task_ids": list(self.ordered_task_ids),
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class AssignmentManifest:
    """Frozen worker ownership for one TaskManifest and one execution wave."""

    task_manifest_sha256: str
    wave_id: str
    node_count: int
    workers_per_node: int
    task_costs: dict[str, int]
    workers: tuple[WorkerAssignment, ...]
    algorithm: str = LPT_ALGORITHM
    metadata: dict[str, Any] | None = None
    schema: str = "hunyuan_video.stage_c_assignment_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema != "hunyuan_video.stage_c_assignment_manifest.v1":
            raise ValueError(f"unsupported AssignmentManifest schema: {self.schema}")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("AssignmentManifest metadata must be a dict")
        require_sha256(self.task_manifest_sha256, "task_manifest_sha256")
        if not self.wave_id.strip():
            raise ValueError("wave_id is required")
        if self.node_count < 1 or self.workers_per_node < 1:
            raise ValueError("node_count and workers_per_node must be positive")
        if self.algorithm != LPT_ALGORITHM:
            raise ValueError(f"unsupported assignment algorithm: {self.algorithm}")
        costs: dict[str, int] = {}
        for task_id, cost in self.task_costs.items():
            if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
                raise ValueError(f"task cost must be a positive integer: {task_id}")
            costs[str(task_id)] = cost
        workers = tuple(self.workers)
        worker_count = self.node_count * self.workers_per_node
        if len(workers) != worker_count:
            raise ValueError("worker assignments do not match execution topology")
        if tuple(worker.worker_id for worker in workers) != tuple(range(worker_count)):
            raise ValueError("worker IDs must be contiguous and ordered")
        assigned = [task_id for worker in workers for task_id in worker.ordered_task_ids]
        if len(assigned) != len(set(assigned)):
            raise ValueError("a task is assigned more than once")
        if set(assigned) != set(costs):
            raise ValueError("assignment task coverage differs from cost table")
        for worker in workers:
            expected = sum(costs[task_id] for task_id in worker.ordered_task_ids)
            if worker.estimated_cost != expected:
                raise ValueError(f"worker {worker.worker_id} estimated cost mismatch")
        object.__setattr__(self, "task_costs", dict(sorted(costs.items())))
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "metadata", json.loads(canonical_json_bytes(self.metadata or {})))

    @property
    def cost_table_sha256(self) -> str:
        return hash_json({"unit": "integer_cost", "task_costs": self.task_costs})

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_manifest_sha256": self.task_manifest_sha256,
            "wave_id": self.wave_id,
            "node_count": self.node_count,
            "workers_per_node": self.workers_per_node,
            "algorithm": self.algorithm,
            "cost_table_sha256": self.cost_table_sha256,
            "task_costs": dict(self.task_costs),
            "workers": [worker.as_dict() for worker in self.workers],
            "metadata": deepcopy(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return hash_json(self.as_dict())

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["manifest_sha256"] = self.sha256
        return payload

    @classmethod
    def from_manifest_dict(cls, payload: dict[str, Any]) -> "AssignmentManifest":
        body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        expected = require_sha256(payload.get("manifest_sha256", ""), "manifest_sha256")
        if hash_json(body) != expected:
            raise ValueError("AssignmentManifest self-hash mismatch")
        declared_cost_hash = require_sha256(body.pop("cost_table_sha256", ""), "cost_table_sha256")
        workers = tuple(
            WorkerAssignment(
                worker_id=int(row["worker_id"]),
                ordered_task_ids=tuple(str(task_id) for task_id in row["ordered_task_ids"]),
                estimated_cost=int(row["estimated_cost"]),
            )
            for row in body.pop("workers")
        )
        manifest = cls(workers=workers, **body)
        if manifest.cost_table_sha256 != declared_cost_hash:
            raise ValueError("AssignmentManifest cost-table hash mismatch")
        return manifest

    @classmethod
    def load(cls, path: Path) -> "AssignmentManifest":
        return cls.from_manifest_dict(json.loads(path.read_text(encoding="utf-8")))

    def tasks_for_worker(self, worker_id: int) -> tuple[str, ...]:
        if not 0 <= worker_id < len(self.workers):
            raise IndexError(f"worker_id out of range: {worker_id}")
        return self.workers[worker_id].ordered_task_ids


def _normalize_costs(task_manifest: TaskManifest, task_costs: Mapping[str, int]) -> dict[str, int]:
    expected_ids = set(task_manifest.task_by_id())
    observed_ids = {str(task_id) for task_id in task_costs}
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(f"cost-table coverage mismatch; missing={missing}, extra={extra}")
    normalized: dict[str, int] = {}
    for task_id, cost in task_costs.items():
        if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
            raise ValueError(f"task cost must be a positive integer: {task_id}")
        normalized[str(task_id)] = cost
    return normalized


def build_lpt_assignment(
    task_manifest: TaskManifest,
    task_costs: Mapping[str, int],
    *,
    node_count: int,
    workers_per_node: int,
    wave_id: str,
    metadata: dict[str, Any] | None = None,
) -> AssignmentManifest:
    """Pack tasks by deterministic LPT using integer costs only."""

    if node_count < 1 or workers_per_node < 1:
        raise ValueError("node_count and workers_per_node must be positive")
    costs = _normalize_costs(task_manifest, task_costs)
    task_by_id = task_manifest.task_by_id()
    ordered_tasks = sorted(
        task_by_id.values(),
        key=lambda item: (
            -costs[item.run_spec.task_id],
            item.formal_row_id,
            item.run_spec.protocol_id,
            item.run_spec.mode,
            item.run_spec.seed,
            item.run_spec.prompt_id,
            item.run_spec.repeat,
            item.run_spec.task_id,
        ),
    )
    worker_count = node_count * workers_per_node
    totals = [0] * worker_count
    task_lists: list[list[str]] = [[] for _ in range(worker_count)]
    for item in ordered_tasks:
        worker_id = min(
            range(worker_count),
            key=lambda index: (totals[index], len(task_lists[index]), index),
        )
        task_id = item.run_spec.task_id
        task_lists[worker_id].append(task_id)
        totals[worker_id] += costs[task_id]
    workers = tuple(
        WorkerAssignment(
            worker_id=worker_id,
            ordered_task_ids=tuple(task_lists[worker_id]),
            estimated_cost=totals[worker_id],
        )
        for worker_id in range(worker_count)
    )
    return AssignmentManifest(
        task_manifest_sha256=task_manifest.sha256,
        wave_id=wave_id,
        node_count=node_count,
        workers_per_node=workers_per_node,
        task_costs=costs,
        workers=workers,
        metadata=metadata,
    )
