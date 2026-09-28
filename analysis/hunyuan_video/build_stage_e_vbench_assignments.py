#!/usr/bin/env python3
"""Build one static production assignment for each Stage E VBench row."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hunyuan_video.assignment import AssignmentManifest, WorkerAssignment
from hunyuan_video.config import TaskManifest, hash_json
from hunyuan_video.records import write_immutable_json


VBENCH_PATH = ROOT / "resources/hunyuan_video/vbench_metadata.v1.json"
STAGE = "E6"
SPLIT = "VBench"
PROTOCOL_ID = "HY-CachePaper-480"
BUDGET_IDS = ("B50", "B70", "B82")
BUDGET_CACHE_COUNTS = {"B50": 25, "B70": 35, "B82": 41}
SEEDS = (42, 43, 44, 45, 46)
ROWS_PER_BUDGET = 5
ARTIFACT_COUNT = 944
TASKS_PER_ROW = ARTIFACT_COUNT * len(SEEDS)
DEFAULT_NODE_COUNT = 20
DEFAULT_WORKERS_PER_NODE = 4
_SAFE_ROW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_vbench_metadata(path: Path = VBENCH_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("VBench metadata must be a JSON object")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if hash_json(body) != payload.get("manifest_sha256"):
        raise ValueError("VBench metadata self-hash mismatch")
    artifacts = payload.get("artifacts")
    if (
        payload.get("schema") != "hunyuan_video.vbench_metadata.v1"
        or payload.get("metadata_row_count") != 946
        or payload.get("exact_unique_prompt_count") != ARTIFACT_COUNT
        or payload.get("canonical_unique_prompt_count") != 943
        or not isinstance(artifacts, list)
        or len(artifacts) != ARTIFACT_COUNT
    ):
        raise ValueError("VBench metadata coverage drift")
    artifact_ids = [item.get("artifact_id") for item in artifacts]
    if any(not isinstance(value, str) or not value for value in artifact_ids):
        raise ValueError("VBench metadata contains an invalid artifact_id")
    if len(set(artifact_ids)) != ARTIFACT_COUNT:
        raise ValueError("VBench artifact IDs are not unique")
    return payload


def validate_stage_e_vbench_manifest(
    manifest: TaskManifest,
    *,
    vbench_metadata: dict[str, Any] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Validate five E6 VBench rows for every active passed budget."""

    vbench = vbench_metadata or load_vbench_metadata()
    vbench_rows = tuple(row for row in manifest.formal_rows if row.stage == STAGE and row.split == SPLIT)
    row_count = len(vbench_rows)
    if row_count not in {5, 10, 15}:
        raise ValueError("Stage E VBench requires five rows per active budget")

    artifacts = vbench["artifacts"]
    artifact_by_id = {item["artifact_id"]: item for item in artifacts}
    expected_artifact_ids = set(artifact_by_id)
    rows = {row.row_id: row for row in vbench_rows}
    if len(rows) != row_count:
        raise ValueError("Stage E VBench row IDs must be unique")
    budget_counts: Counter[str] = Counter()
    rows_by_budget: dict[str, list[Any]] = defaultdict(list)
    for row in rows.values():
        if not _SAFE_ROW_ID.fullmatch(row.row_id):
            raise ValueError(f"unsafe Stage E VBench row_id: {row.row_id!r}")
        if (
            row.stage != STAGE
            or row.split != SPLIT
            or row.protocol_id != PROTOCOL_ID
            or row.mode == "original"
            or row.seeds != SEEDS
            or row.repeats != 1
            or row.budget_id not in BUDGET_IDS
            or row.reference_row_id is not None
            or row.source_protocol_id is not None
        ):
            raise ValueError(f"Stage E VBench candidate row identity drift: {row.row_id}")
        budget_counts[str(row.budget_id)] += 1
        rows_by_budget[str(row.budget_id)].append(row)
    active_budgets = set(budget_counts)
    if (
        not active_budgets
        or not active_budgets.issubset(BUDGET_IDS)
        or any(count != ROWS_PER_BUDGET for count in budget_counts.values())
    ):
        raise ValueError("Stage E VBench requires five candidate rows per budget")
    for budget_id, budget_rows in rows_by_budget.items():
        modes = Counter(row.mode for row in budget_rows)
        if modes != Counter({"seacache": 1, "golden_reuse_schedule": 4}):
            raise ValueError(f"Stage E VBench {budget_id} requires one Sea and four fixed-reuse rows")
        cache_count = BUDGET_CACHE_COUNTS[budget_id]
        for row in budget_rows:
            if row.mode != "golden_reuse_schedule":
                continue
            steps = row.method_config.get("cache_steps")
            if (
                row.method_config.get("cache_count") != cache_count
                or not isinstance(steps, list)
                or any(isinstance(step, bool) or not isinstance(step, int) for step in steps)
                or len(steps) != cache_count
                or steps != sorted(set(steps))
                or steps[0] < 1
                or steps[-1] > 48
            ):
                raise ValueError(f"Stage E VBench fixed schedule drift: {row.row_id}")

    tasks_by_row: dict[str, list[Any]] = defaultdict(list)
    task_ids: set[str] = set()
    vbench_tasks = tuple(task for task in manifest.tasks if task.formal_row_id in rows)
    expected_task_count = row_count * TASKS_PER_ROW
    if len(vbench_tasks) != expected_task_count:
        raise ValueError(f"Stage E VBench requires exactly {expected_task_count} tasks")
    for task in vbench_tasks:
        run = task.run_spec
        row = rows.get(task.formal_row_id)
        if row is None:
            raise ValueError(f"Stage E VBench task references an unknown row: {task.formal_row_id}")
        if run.task_id in task_ids:
            raise ValueError(f"duplicate Stage E VBench task_id: {run.task_id}")
        task_ids.add(run.task_id)
        artifact = artifact_by_id.get(run.prompt_id)
        if artifact is None or run.prompt != artifact.get("prompt"):
            raise ValueError(f"Stage E VBench task/artifact mismatch: {run.task_id}")
        if (
            run.phase != STAGE
            or run.protocol_id != PROTOCOL_ID
            or run.mode != row.mode
            or run.seed not in SEEDS
            or run.repeat != 0
            or run.method_config != row.method_config
        ):
            raise ValueError(f"Stage E VBench task identity drift: {run.task_id}")
        tasks_by_row[row.row_id].append(task)

    frozen: dict[str, tuple[Any, ...]] = {}
    expected_pairs = {(artifact_id, seed) for artifact_id in expected_artifact_ids for seed in SEEDS}
    for row_id in sorted(rows, key=lambda value: (rows[value].budget_id, value)):
        tasks = tasks_by_row.get(row_id, [])
        observed_pairs = {(task.run_spec.prompt_id, task.run_spec.seed) for task in tasks}
        if len(tasks) != TASKS_PER_ROW or observed_pairs != expected_pairs:
            raise ValueError(f"Stage E VBench row coverage drift: {row_id}")
        frozen[row_id] = tuple(
            sorted(
                tasks,
                key=lambda task: (
                    task.run_spec.seed,
                    task.run_spec.prompt_id,
                    task.run_spec.task_id,
                ),
            )
        )
    if set(frozen) != set(rows):
        raise ValueError("Stage E VBench task coverage omits a candidate row")
    return frozen


def _workers(tasks: Iterable[Any], count: int) -> tuple[WorkerAssignment, ...]:
    if count < 1:
        raise ValueError("worker count must be positive")
    assigned: list[list[str]] = [[] for _ in range(count)]
    for index, task in enumerate(tasks):
        assigned[index % count].append(task.run_spec.task_id)
    return tuple(
        WorkerAssignment(
            worker_id=worker_id,
            ordered_task_ids=tuple(task_ids),
            estimated_cost=len(task_ids),
        )
        for worker_id, task_ids in enumerate(assigned)
    )


def build_assignments(
    manifest: TaskManifest,
    *,
    node_count: int = DEFAULT_NODE_COUNT,
    workers_per_node: int = DEFAULT_WORKERS_PER_NODE,
    vbench_metadata: dict[str, Any] | None = None,
) -> list[tuple[str, AssignmentManifest]]:
    if node_count < 1 or workers_per_node < 1:
        raise ValueError("node_count and workers_per_node must be positive")
    vbench = vbench_metadata or load_vbench_metadata()
    tasks_by_row = validate_stage_e_vbench_manifest(manifest, vbench_metadata=vbench)
    rows = {row.row_id: row for row in manifest.formal_rows}
    worker_count = node_count * workers_per_node
    manifest_sha256 = manifest.sha256
    outputs: list[tuple[str, AssignmentManifest]] = []
    for index, (row_id, tasks) in enumerate(tasks_by_row.items()):
        task_costs = {task.run_spec.task_id: 1 for task in tasks}
        assignment = AssignmentManifest(
            task_manifest_sha256=manifest_sha256,
            wave_id=f"e6-vbench-row{index:02d}-{row_id}",
            node_count=node_count,
            workers_per_node=workers_per_node,
            task_costs=task_costs,
            workers=_workers(tasks, worker_count),
            metadata={
                "artifact_count": ARTIFACT_COUNT,
                "budget_id": rows[row_id].budget_id,
                "row_id": row_id,
                "seed_count": len(SEEDS),
                "seeds": list(SEEDS),
                "stage": STAGE,
                "task_count": TASKS_PER_ROW,
                "vbench_metadata_sha256": vbench["manifest_sha256"],
            },
        )
        outputs.append((f"e6_vbench_row{index:02d}_{row_id}.json", assignment))
    if len(outputs) != len(tasks_by_row):
        raise AssertionError("Stage E VBench assignment count drift")
    covered = {task_id for _, assignment in outputs for task_id in assignment.task_costs}
    expected = {task.run_spec.task_id for task in manifest.tasks if task.formal_row_id in tasks_by_row}
    if covered != expected:
        raise AssertionError("Stage E VBench assignments do not cover all VBench tasks")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node-count", type=int, default=DEFAULT_NODE_COUNT)
    parser.add_argument("--workers-per-node", type=int, default=DEFAULT_WORKERS_PER_NODE)
    args = parser.parse_args()
    manifest = TaskManifest.load(args.task_manifest)
    outputs = build_assignments(
        manifest,
        node_count=args.node_count,
        workers_per_node=args.workers_per_node,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, assignment in outputs:
        write_immutable_json(args.output_dir / name, assignment.to_manifest_dict())
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "row_count": len(outputs),
                "task_count": sum(len(assignment.task_costs) for _, assignment in outputs),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
