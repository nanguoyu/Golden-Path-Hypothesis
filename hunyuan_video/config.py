from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS_PATH = ROOT / "resources/hunyuan_video/generation_protocols.v1.json"
SOURCE_GENERATION_PROTOCOLS_PATH = (
    ROOT / "resources/hunyuan_video/source_generation_protocols.v1.json"
)
DATASET_PATH = ROOT / "resources/hunyuan_video/dataset_splits.v2.json"
VAE_TILING_ENABLED = True


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_sha256(value: str, field_name: str) -> str:
    value = str(value)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _value(section: dict[str, Any], key: str) -> Any:
    return section[key]["value"]


@dataclass(frozen=True)
class GenerationProtocol:
    protocol_id: str
    role: str
    height: int
    width: int
    frames: int
    fps: int
    steps: int
    flow_shift: float
    flow_solver: str
    flow_reverse: bool
    guidance_scale: float
    embedded_guidance_scale: float
    cpu_offload: bool
    vae_tiling: bool
    precision: str
    vae_precision: str
    text_encoder_precision: str
    text_encoder_precision_2: str
    model: str
    checkpoint: str
    load_key: str
    prompt_template_video: str
    hidden_state_skip_layer: int
    apply_final_norm: bool
    source_commit: str
    manifest_sha256: str
    protocol_sha256: str


def load_protocol_registry(protocol_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    found: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in (PROTOCOLS_PATH, SOURCE_GENERATION_PROTOCOLS_PATH):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        if hash_json(body) != payload.get("manifest_sha256"):
            raise ValueError(f"generation protocol manifest self-hash mismatch: {path}")
        found.extend(
            (payload, row)
            for row in payload.get("protocols", [])
            if row.get("protocol_id") == protocol_id
        )
    if len(found) != 1:
        raise KeyError(f"unknown or ambiguous generation protocol: {protocol_id}")
    return found[0]


def load_protocol(protocol_id: str) -> GenerationProtocol:
    payload, row = load_protocol_registry(protocol_id)
    generation = row["generation"]
    sampler = row["sampler"]
    precision = row["precision"]
    model = row["model_identity"]
    prompt = row["prompt_processing"]
    runtime = row["runtime_topology"]
    return GenerationProtocol(
        protocol_id=protocol_id,
        role=row["role"],
        height=int(_value(generation, "height")),
        width=int(_value(generation, "width")),
        frames=int(_value(generation, "frames")),
        fps=int(_value(generation, "fps")),
        steps=int(_value(generation, "steps")),
        flow_shift=float(_value(sampler, "flow_shift")),
        flow_solver=str(_value(sampler, "solver")),
        flow_reverse=bool(_value(sampler, "flow_reverse")),
        guidance_scale=float(_value(sampler, "guidance_scale")),
        embedded_guidance_scale=float(_value(sampler, "embedded_guidance_scale")),
        cpu_offload=bool(_value(runtime, "cpu_offload")),
        vae_tiling=VAE_TILING_ENABLED,
        precision=str(_value(precision, "dit")),
        vae_precision=str(_value(precision, "vae")),
        text_encoder_precision=str(_value(precision, "text_encoder_1")),
        text_encoder_precision_2=str(_value(precision, "text_encoder_2")),
        model=str(_value(model, "model")),
        checkpoint=str(_value(model, "checkpoint")),
        load_key=str(_value(model, "load_key")),
        prompt_template_video=str(_value(prompt, "prompt_template")),
        hidden_state_skip_layer=int(_value(prompt, "hidden_state_skip_layer")),
        apply_final_norm=bool(_value(prompt, "apply_final_norm")),
        source_commit=str(_value(model, "source_commit")),
        manifest_sha256=payload["manifest_sha256"],
        protocol_sha256=hash_json(row),
    )


@dataclass(frozen=True)
class RunSpec:
    phase: str
    task_id: str
    protocol_id: str
    mode: str
    prompt_id: str
    prompt: str
    seed: int
    repeat: int
    method_config: dict[str, Any] = field(default_factory=dict)
    schema: str = "hunyuan_video.run_spec.v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hash_json(self.as_dict())


@dataclass(frozen=True)
class FormalRow:
    """Immutable Stage C row identity, independent of task partitioning."""

    row_id: str
    stage: str
    protocol_id: str
    mode: str
    split: str
    seeds: tuple[int, ...]
    repeats: int
    method_config: dict[str, Any] = field(default_factory=dict)
    budget_id: str | None = None
    reference_row_id: str | None = None
    source_protocol_id: str | None = None
    schema: str = "hunyuan_video.formal_row.v1"

    def __post_init__(self) -> None:
        if self.schema != "hunyuan_video.formal_row.v1":
            raise ValueError(f"unsupported FormalRow schema: {self.schema}")
        for name in ("row_id", "stage", "protocol_id", "mode", "split"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"FormalRow {name} is required")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("FormalRow seeds must be non-empty and unique")
        if int(self.repeats) < 1:
            raise ValueError("FormalRow repeats must be positive")
        if not isinstance(self.method_config, dict):
            raise TypeError("FormalRow method_config must be a dict")
        # Detach identity-bearing JSON from caller-owned mutable objects.
        normalized_config = json.loads(canonical_json_bytes(self.method_config))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "repeats", int(self.repeats))
        object.__setattr__(self, "method_config", normalized_config)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "stage": self.stage,
            "protocol_id": self.protocol_id,
            "mode": self.mode,
            "split": self.split,
            "seeds": list(self.seeds),
            "repeats": self.repeats,
            "method_config": deepcopy(self.method_config),
            "budget_id": self.budget_id,
            "reference_row_id": self.reference_row_id,
            "source_protocol_id": self.source_protocol_id,
            "schema": self.schema,
        }

    @property
    def sha256(self) -> str:
        return hash_json(self.as_dict())

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["row_sha256"] = self.sha256
        return payload

    @classmethod
    def from_manifest_dict(cls, payload: dict[str, Any]) -> "FormalRow":
        body = {key: value for key, value in payload.items() if key != "row_sha256"}
        expected = require_sha256(payload.get("row_sha256", ""), "row_sha256")
        if hash_json(body) != expected:
            raise ValueError("FormalRow self-hash mismatch")
        return cls(**body)


@dataclass(frozen=True)
class FormalTask:
    """A Stage C task linked to exactly one formal row."""

    formal_row_id: str
    run_spec: RunSpec

    def __post_init__(self) -> None:
        if not self.formal_row_id or not self.run_spec.task_id:
            raise ValueError("FormalTask row and task IDs are required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "formal_row_id": self.formal_row_id,
            "run_spec": self.run_spec.as_dict(),
            "run_spec_sha256": self.run_spec.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormalTask":
        run_spec = RunSpec(**payload["run_spec"])
        expected = require_sha256(payload.get("run_spec_sha256", ""), "run_spec_sha256")
        if run_spec.sha256 != expected:
            raise ValueError("RunSpec self-hash mismatch")
        return cls(formal_row_id=str(payload["formal_row_id"]), run_spec=run_spec)


@dataclass(frozen=True)
class TaskManifest:
    """Self-hashed formal task set; assignments bind this manifest by hash."""

    formal_rows: tuple[FormalRow, ...]
    tasks: tuple[FormalTask, ...]
    dataset_manifest_sha256: str
    protocol_manifest_sha256s: dict[str, str]
    parent_manifest_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "hunyuan_video.stage_c_task_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema != "hunyuan_video.stage_c_task_manifest.v1":
            raise ValueError(f"unsupported TaskManifest schema: {self.schema}")
        if not isinstance(self.metadata, dict):
            raise TypeError("TaskManifest metadata must be a dict")
        rows = tuple(sorted(self.formal_rows, key=lambda row: row.row_id))
        tasks = tuple(
            sorted(
                self.tasks,
                key=lambda task: (
                    task.formal_row_id,
                    task.run_spec.protocol_id,
                    task.run_spec.mode,
                    task.run_spec.seed,
                    task.run_spec.prompt_id,
                    task.run_spec.repeat,
                    task.run_spec.task_id,
                ),
            )
        )
        require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        protocol_hashes = {
            str(protocol_id): require_sha256(digest, f"protocol_manifest_sha256s[{protocol_id}]")
            for protocol_id, digest in self.protocol_manifest_sha256s.items()
        }
        if self.parent_manifest_sha256 is not None:
            require_sha256(self.parent_manifest_sha256, "parent_manifest_sha256")
        if not rows or not tasks:
            raise ValueError("TaskManifest formal_rows and tasks must be non-empty")
        row_by_id = {row.row_id: row for row in rows}
        if len(row_by_id) != len(rows):
            raise ValueError("TaskManifest has duplicate formal row IDs")
        if set(protocol_hashes) != {row.protocol_id for row in rows}:
            raise ValueError("protocol manifest hashes must exactly cover formal-row protocols")
        task_ids = [task.run_spec.task_id for task in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("TaskManifest has duplicate task IDs")
        used_rows: set[str] = set()
        for task in tasks:
            row = row_by_id.get(task.formal_row_id)
            if row is None:
                raise ValueError(f"task references unknown formal row: {task.formal_row_id}")
            run = task.run_spec
            if run.phase != row.stage or run.protocol_id != row.protocol_id or run.mode != row.mode:
                raise ValueError(f"task identity differs from formal row: {run.task_id}")
            if run.seed not in row.seeds or not 0 <= run.repeat < row.repeats:
                raise ValueError(f"task seed/repeat differs from formal row: {run.task_id}")
            if canonical_json_bytes(run.method_config) != canonical_json_bytes(row.method_config):
                raise ValueError(f"task method_config differs from formal row: {run.task_id}")
            used_rows.add(row.row_id)
        if used_rows != set(row_by_id):
            raise ValueError("every formal row must own at least one task")
        object.__setattr__(self, "formal_rows", rows)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "protocol_manifest_sha256s", protocol_hashes)
        object.__setattr__(self, "metadata", json.loads(canonical_json_bytes(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "protocol_manifest_sha256s": dict(sorted(self.protocol_manifest_sha256s.items())),
            "formal_rows": [row.to_manifest_dict() for row in self.formal_rows],
            "tasks": [task.as_dict() for task in self.tasks],
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
    def from_manifest_dict(cls, payload: dict[str, Any]) -> "TaskManifest":
        body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        expected = require_sha256(payload.get("manifest_sha256", ""), "manifest_sha256")
        if hash_json(body) != expected:
            raise ValueError("TaskManifest self-hash mismatch")
        rows = tuple(FormalRow.from_manifest_dict(row) for row in body.pop("formal_rows"))
        tasks = tuple(FormalTask.from_dict(task) for task in body.pop("tasks"))
        return cls(formal_rows=rows, tasks=tasks, **body)

    @classmethod
    def load(cls, path: Path) -> "TaskManifest":
        return cls.from_manifest_dict(json.loads(path.read_text(encoding="utf-8")))

    def task_by_id(self) -> dict[str, FormalTask]:
        return {task.run_spec.task_id: task for task in self.tasks}


def load_stage_b_tasks(path: Path) -> list[RunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if hash_json(body) != payload.get("manifest_sha256"):
        raise ValueError("Stage B task manifest self-hash mismatch")
    return [RunSpec(**row) for row in payload["tasks"]]
