#!/usr/bin/env python3
"""Build deterministic VBench identity and confirmatory audit manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path("reference/vbench/code/vbench/VBench_full_info.json")
METADATA_PATH = Path("resources/hunyuan_video/vbench_metadata.v1.json")
AUDIT_PATH = Path("resources/hunyuan_video/vbench_audit_sample.v1.json")
PROTOCOLS_PATH = Path("resources/hunyuan_video/generation_protocols.v1.json")
VBENCH_COMMIT = "45e79ec14e69a2187202c675d2dbce1a71843d53"
SOURCE_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
AUDIT_SEEDS = (42, 43, 44, 45, 46)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_json(value: Any) -> str:
    return hash_bytes(canonical_json_bytes(value))


def normalize_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).lower().split())


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_without_duplicates,
        parse_constant=_reject_constant,
    )


def build_vbench_metadata(root: Path = ROOT) -> dict[str, Any]:
    source_path = root / SOURCE_PATH
    if hash_bytes(source_path.read_bytes()) != SOURCE_SHA256:
        raise ValueError("VBench source SHA256 mismatch")
    source = _load_strict_json(source_path)
    if not isinstance(source, list) or len(source) != 946:
        raise ValueError("expected 946 VBench metadata rows")

    schema_counts = defaultdict(int)
    raw_prompts: list[str] = []
    canonical_prompts: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    dimensions: set[str] = set()

    for row_index, row in enumerate(source):
        if not isinstance(row, dict):
            raise ValueError(f"VBench row {row_index} is not an object")
        keys = frozenset(row)
        allowed = {
            frozenset(("prompt_en", "dimension")),
            frozenset(("prompt_en", "dimension", "auxiliary_info")),
        }
        if keys not in allowed:
            raise ValueError(f"unexpected VBench row schema at {row_index}: {sorted(keys)}")
        schema_counts[len(keys)] += 1
        prompt = row["prompt_en"]
        row_dimensions = row["dimension"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"invalid VBench prompt at row {row_index}")
        if (
            not isinstance(row_dimensions, list)
            or not row_dimensions
            or any(not isinstance(value, str) or not value for value in row_dimensions)
        ):
            raise ValueError(f"invalid VBench dimensions at row {row_index}")
        artifact_id = hash_bytes(prompt.encode("utf-8"))
        canonical_prompt = normalize_prompt(prompt)
        canonical_sha256 = hash_bytes(canonical_prompt.encode("utf-8"))
        raw_prompts.append(prompt)
        canonical_prompts.append(canonical_prompt)
        grouped_indices[prompt].append(row_index)
        dimensions.update(row_dimensions)
        metadata_rows.append(
            {
                "artifact_id": artifact_id,
                "canonical_sha256": canonical_sha256,
                "metadata": row,
                "row_index": row_index,
            }
        )

    if dict(schema_counts) != {2: 440, 3: 506}:
        raise ValueError(f"unexpected VBench row schema counts: {dict(schema_counts)}")

    artifacts = []
    for prompt, row_indices in grouped_indices.items():
        canonical_prompt = normalize_prompt(prompt)
        artifacts.append(
            {
                "artifact_id": hash_bytes(prompt.encode("utf-8")),
                "canonical_prompt": canonical_prompt,
                "canonical_sha256": hash_bytes(canonical_prompt.encode("utf-8")),
                "metadata_row_indices": row_indices,
                "prompt": prompt,
            }
        )
    artifacts.sort(key=lambda item: item["artifact_id"])

    if len(artifacts) != 944 or len(set(canonical_prompts)) != 943:
        raise ValueError("expected 944 exact artifacts and 943 canonical prompts")
    if sorted(dimensions) != [
        "aesthetic_quality",
        "appearance_style",
        "background_consistency",
        "color",
        "dynamic_degree",
        "human_action",
        "imaging_quality",
        "motion_smoothness",
        "multiple_objects",
        "object_class",
        "overall_consistency",
        "scene",
        "spatial_relationship",
        "subject_consistency",
        "temporal_flickering",
        "temporal_style",
    ]:
        raise ValueError("unexpected VBench dimensions")

    manifest: dict[str, Any] = {
        "artifact_prompts_sha256": hash_json(artifacts),
        "artifacts": artifacts,
        "canonical_unique_prompt_count": len(set(canonical_prompts)),
        "dimensions": sorted(dimensions),
        "exact_unique_prompt_count": len(artifacts),
        "metadata_identity_sha256": hash_json(metadata_rows),
        "metadata_row_count": len(source),
        "metadata_rows": metadata_rows,
        "normalization": "NFKC-lower-whitespace-collapse.v1",
        "schema": "hunyuan_video.vbench_metadata.v1",
        "source": {
            "file_sha256": SOURCE_SHA256,
            "path": SOURCE_PATH.as_posix(),
            "repository_commit": VBENCH_COMMIT,
        },
        "source_canonical_prompts_sha256": hash_json(canonical_prompts),
        "source_metadata_sha256": hash_json(source),
        "source_raw_prompts_sha256": hash_json(raw_prompts),
    }
    manifest["manifest_sha256"] = hash_json(manifest)
    return manifest


def _protocol_identity(root: Path) -> dict[str, str]:
    payload = _load_strict_json(root / PROTOCOLS_PATH)
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if hash_json(body) != payload.get("manifest_sha256"):
        raise ValueError("generation protocols self-hash mismatch")
    matches = [
        protocol
        for protocol in payload["protocols"]
        if protocol["protocol_id"] == "HY-CachePaper-480"
    ]
    if len(matches) != 1:
        raise ValueError("missing HY-CachePaper-480 protocol")
    return {
        "manifest_path": PROTOCOLS_PATH.as_posix(),
        "manifest_sha256": payload["manifest_sha256"],
        "protocol_id": "HY-CachePaper-480",
        "protocol_sha256": hash_json(matches[0]),
    }


def build_vbench_audit_sample(
    metadata: dict[str, Any] | None = None, root: Path = ROOT
) -> dict[str, Any]:
    if metadata is None:
        metadata = build_vbench_metadata(root)
    rows = metadata["metadata_rows"]
    artifact_dimensions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        artifact_dimensions[row["artifact_id"]].update(row["metadata"]["dimension"])
    artifacts = {item["artifact_id"]: item for item in metadata["artifacts"]}

    selected: set[str] = set()
    items = []
    for dimension in metadata["dimensions"]:
        candidates = sorted(
            artifact_id
            for artifact_id, observed in artifact_dimensions.items()
            if dimension in observed and artifact_id not in selected
        )
        if not candidates:
            raise ValueError(f"no unique audit witness for dimension {dimension}")
        artifact_id = candidates[0]
        selected.add(artifact_id)
        artifact = artifacts[artifact_id]
        items.append(
            {
                "artifact_id": artifact_id,
                "canonical_sha256": artifact["canonical_sha256"],
                "metadata_row_indices": artifact["metadata_row_indices"],
                "observed_dimensions": sorted(artifact_dimensions[artifact_id]),
                "primary_dimension": dimension,
                "raw_sha256": artifact_id,
            }
        )

    artifact_ids = [item["artifact_id"] for item in items]
    manifest: dict[str, Any] = {
        "algorithm": "dimension-lexicographic-unique-v1",
        "artifact_count": len(items),
        "artifact_ids_sha256": hash_json(artifact_ids),
        "claim_scope": "raw-media evaluator conformance only; not a VBench population estimator or method-selection set",
        "coverage_count": len({item["primary_dimension"] for item in items}),
        "dimension_order": list(metadata["dimensions"]),
        "dimension_to_artifact": {
            item["primary_dimension"]: item["artifact_id"] for item in items
        },
        "items": items,
        "protocol": _protocol_identity(root),
        "retention_deadline": "candidate/reference pairs retained through the corresponding Gate D or Gate F signature",
        "retention_scope": "every formal VBench row x every fixed seed retains the same paired candidate/reference artifact set",
        "schema": "hunyuan_video.vbench_audit_sample.v1",
        "seeds": list(AUDIT_SEEDS),
        "source": {
            "metadata_manifest_path": METADATA_PATH.as_posix(),
            "metadata_manifest_sha256": metadata["manifest_sha256"],
            "vbench_repository_commit": VBENCH_COMMIT,
            "vbench_source_file_sha256": SOURCE_SHA256,
        },
    }
    manifest["manifest_sha256"] = hash_json(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--metadata-output", type=Path, default=ROOT / METADATA_PATH)
    parser.add_argument("--audit-output", type=Path, default=ROOT / AUDIT_PATH)
    args = parser.parse_args()
    metadata = build_vbench_metadata(args.root)
    audit = build_vbench_audit_sample(metadata, args.root)
    args.metadata_output.write_bytes(canonical_json_bytes(metadata) + b"\n")
    args.audit_output.write_bytes(canonical_json_bytes(audit) + b"\n")


if __name__ == "__main__":
    main()
