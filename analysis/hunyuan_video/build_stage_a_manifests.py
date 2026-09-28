#!/usr/bin/env python3
"""Build the deterministic HunyuanVideo Stage A dataset split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path("reference/hunyuan_video/code/assets/PenguinVideoBenchmark.csv")
METADATA_PATH = Path("resources/hunyuan_video/vbench_metadata.v1.json")
AUDIT_PATH = Path("resources/hunyuan_video/vbench_audit_sample.v1.json")
OUTPUT_PATH = Path("resources/hunyuan_video/dataset_splits.v2.json")
TENCENT_COMMIT = "e748c73ac064728bf6bd15b1cdb8161e55a4f331"
PRIMARY_SPLITS = (
    ("B-CAL48", 48),
    ("B-TRAIN48", 48),
    ("GP-S128", 128),
    ("GP-V128", 128),
    ("BASE119", 119),
    ("GP-C128", 128),
)
PRIMARY_POLICIES = {
    "B-CAL48": ("gate-and-compute-calibration-only", False, False),
    "B-TRAIN48": ("reserved-trainable-extension", False, False),
    "GP-S128": ("golden-path-search", True, False),
    "GP-V128": ("golden-path-validation-and-ranking", True, False),
    "BASE119": ("core-baseline-fidelity", False, True),
    "GP-C128": ("unseen-confirmation", False, True),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def normalize_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).lower().split())


def _load_source(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["", "prompt"]:
            raise ValueError(f"unexpected CSV columns: {reader.fieldnames!r}")
        rows = list(reader)

    if len(rows) != 600:
        raise ValueError(f"expected 600 source rows, found {len(rows)}")
    if any(row[""] != str(index) for index, row in enumerate(rows)):
        raise ValueError("source indices must be contiguous 0..599")

    raw_prompts = [row["prompt"] for row in rows]
    canonical_prompts = [normalize_prompt(prompt) for prompt in raw_prompts]
    if len(set(raw_prompts)) != 599 or len(set(canonical_prompts)) != 599:
        raise ValueError("expected 599 exact and canonical unique prompts")

    grouped: dict[str, list[int]] = defaultdict(list)
    for source_index, canonical_prompt in enumerate(canonical_prompts):
        grouped[canonical_prompt].append(source_index)

    prompts = []
    for canonical_prompt, source_indices in grouped.items():
        raw_values = {raw_prompts[index] for index in source_indices}
        if len(raw_values) != 1:
            raise ValueError("canonical duplicates must also be exact duplicates")
        canonical_sha256 = sha256_bytes(canonical_prompt.encode("utf-8"))
        prompts.append(
            {
                "canonical_prompt": canonical_prompt,
                "canonical_sha256": canonical_sha256,
                "prompt": raw_prompts[source_indices[0]],
                "prompt_id": canonical_sha256,
                "raw_sha256": sha256_bytes(raw_prompts[source_indices[0]].encode("utf-8")),
                "source_indices": source_indices,
            }
        )

    prompts.sort(
        key=lambda item: (
            item["canonical_sha256"],
            item["canonical_prompt"],
            item["prompt"],
            item["source_indices"],
        )
    )
    if len({item["prompt_id"] for item in prompts}) != 599:
        raise ValueError("canonical SHA256 collision")
    return raw_prompts, prompts


def _split(
    prompt_ids: list[str],
    *,
    role: str,
    candidate_selection_allowed: bool,
    formal_quality_conclusion_allowed: bool,
    parent: str | None = None,
) -> dict[str, Any]:
    split = {
        "candidate_selection_allowed": candidate_selection_allowed,
        "count": len(prompt_ids),
        "formal_quality_conclusion_allowed": formal_quality_conclusion_allowed,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": hash_json(prompt_ids),
        "role": role,
    }
    if parent is not None:
        split["parent"] = parent
    return split


def _metadata_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    without_self = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if hash_json(without_self) != payload.get("manifest_sha256"):
        raise ValueError("VBench metadata self hash is invalid")
    if payload.get("metadata_row_count") != 946 or len(payload.get("metadata_rows", [])) != 946:
        raise ValueError("expected 946 VBench metadata rows")
    if payload.get("exact_unique_prompt_count") != 944 or len(payload.get("artifacts", [])) != 944:
        raise ValueError("expected 944 exact VBench artifacts")
    if len(payload.get("dimensions", [])) != 16:
        raise ValueError("expected 16 VBench dimensions")
    return {
        "canonical_artifact_count": payload["canonical_unique_prompt_count"],
        "dimension_count": 16,
        "exact_artifact_count": 944,
        "file_sha256": sha256_bytes(path.read_bytes()),
        "manifest_sha256": payload["manifest_sha256"],
        "metadata_identity_sha256": payload["metadata_identity_sha256"],
        "metadata_row_count": 946,
        "path": METADATA_PATH.as_posix(),
        "schema": payload["schema"],
    }


def _audit_reference(path: Path, metadata_manifest_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    without_self = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if hash_json(without_self) != payload.get("manifest_sha256"):
        raise ValueError("VBench audit sample self hash is invalid")
    if payload.get("artifact_count") != 16 or payload.get("coverage_count") != 16:
        raise ValueError("expected a 16-artifact/16-dimension VBench audit sample")
    if payload.get("source", {}).get("metadata_manifest_sha256") != metadata_manifest_sha256:
        raise ValueError("VBench audit sample metadata binding mismatch")
    return {
        "algorithm": payload["algorithm"],
        "artifact_count": payload["artifact_count"],
        "artifact_ids_sha256": payload["artifact_ids_sha256"],
        "coverage_count": payload["coverage_count"],
        "file_sha256": sha256_bytes(path.read_bytes()),
        "manifest_sha256": payload["manifest_sha256"],
        "path": AUDIT_PATH.as_posix(),
        "schema": payload["schema"],
        "seeds": payload["seeds"],
    }


def build_manifest(root: Path = ROOT) -> dict[str, Any]:
    source_path = root / SOURCE_PATH
    raw_prompts, prompts = _load_source(source_path)
    canonical_prompts = [normalize_prompt(prompt) for prompt in raw_prompts]
    prompt_ids = [prompt["prompt_id"] for prompt in prompts]

    splits: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, count in PRIMARY_SPLITS:
        role, select, quality = PRIMARY_POLICIES[name]
        splits[name] = _split(
            prompt_ids[offset : offset + count],
            role=role,
            candidate_selection_allowed=select,
            formal_quality_conclusion_allowed=quality,
        )
        offset += count
    if offset != len(prompt_ids):
        raise ValueError("primary split sizes do not exhaust unique prompts")

    for count in (16, 32, 64):
        splits[f"GP-S{count}"] = _split(
            splits["GP-S128"]["prompt_ids"][:count],
            role="successive-halving-prefix",
            candidate_selection_allowed=True,
            formal_quality_conclusion_allowed=False,
            parent="GP-S128",
        )
    splits["T4"] = _split(
        splits["BASE119"]["prompt_ids"][:4],
        role="engineering-pilot-only",
        candidate_selection_allowed=False,
        formal_quality_conclusion_allowed=False,
        parent="BASE119",
    )
    splits["T16"] = _split(
        splits["BASE119"]["prompt_ids"][:16],
        role="timing-only",
        candidate_selection_allowed=False,
        formal_quality_conclusion_allowed=False,
        parent="BASE119",
    )
    splits["AUDIT16"] = _split(
        splits["BASE119"]["prompt_ids"][:16],
        role="confirmatory-raw-media-audit-only",
        candidate_selection_allowed=False,
        formal_quality_conclusion_allowed=False,
        parent="BASE119",
    )
    splits["GP-C-AUDIT16"] = _split(
        splits["GP-C128"]["prompt_ids"][:16],
        role="golden-path-confirmatory-raw-media-audit-only",
        candidate_selection_allowed=False,
        formal_quality_conclusion_allowed=False,
        parent="GP-C128",
    )
    splits["GP-C64"] = _split(
        splits["GP-C128"]["prompt_ids"][:64],
        role="resolution-transfer-only",
        candidate_selection_allowed=False,
        formal_quality_conclusion_allowed=False,
        parent="GP-C128",
    )

    metadata_reference = _metadata_reference(root / METADATA_PATH)
    manifest: dict[str, Any] = {
        "canonical_unique_prompt_count": len(prompts),
        "exact_unique_prompt_count": len(set(raw_prompts)),
        "normalization": "NFKC-lower-whitespace-collapse.v1",
        "ordering": "canonical_sha256,canonical_prompt,prompt,source_indices",
        "primary_split_order": [name for name, _ in PRIMARY_SPLITS],
        "prompt_transport": "structured-json-only; embedded-newlines-preserved",
        "prompts": prompts,
        "prompts_sha256": hash_json(prompts),
        "schema": "hunyuan_video.dataset_splits.v2",
        "source": {
            "file_sha256": sha256_bytes(source_path.read_bytes()),
            "path": SOURCE_PATH.as_posix(),
            "repository_commit": TENCENT_COMMIT,
            "row_count": len(raw_prompts),
        },
        "source_canonical_prompts_sha256": hash_json(canonical_prompts),
        "source_raw_prompts_sha256": hash_json(raw_prompts),
        "splits": splits,
        "partition_sha256": hash_json(
            [{"name": name, "prompt_ids": splits[name]["prompt_ids"]} for name, _ in PRIMARY_SPLITS]
        ),
        "vbench_audit_sample": _audit_reference(
            root / AUDIT_PATH, metadata_reference["manifest_sha256"]
        ),
        "vbench_metadata": metadata_reference,
    }
    manifest["manifest_sha256"] = hash_json(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_PATH)
    args = parser.parse_args()
    args.output.write_bytes(canonical_json_bytes(build_manifest()) + b"\n")


if __name__ == "__main__":
    main()
