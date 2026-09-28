#!/usr/bin/env python3
"""Build deterministic Stage A HunyuanVideo protocol manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GENERATION_OUTPUT_PATH = Path("resources/hunyuan_video/generation_protocols.v1.json")
SOURCE_GENERATION_OUTPUT_PATH = Path(
    "resources/hunyuan_video/source_generation_protocols.v1.json"
)
SOURCE_OUTPUT_PATH = Path("resources/hunyuan_video/source_protocols.v1.json")

STAGE_A_PARENT_COMMIT = "18bdeb6b2d8229d2af5464016ebcd0763897de2d"
REVISIONS = {
    "hicache": "94e11b1a5e7d2b42813ec3fc479339da6fc49207",
    "hunyuan_video": "e748c73ac064728bf6bd15b1cdb8161e55a4f331",
    "seacache": "3b1c688f8d320096d137005d9dbaaa3798ac84b0",
    "taylorseer": "704ee98c74f7f04da443daa3c0aa2cc7803d86e3",
    "teacache": "7c10efc4702c6b619f47805f7abe4a7a08085aa0",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def evidence(value: Any, status: str, source_ref: str) -> dict[str, Any]:
    return {"evidence_status": status, "source_ref": source_ref, "value": value}


def unknown(source_ref: str) -> dict[str, Any]:
    return evidence(None, "unknown", source_ref)


def deviation(kind: str, detail: str, source_ref: str) -> dict[str, str]:
    return {"detail": detail, "kind": kind, "source_ref": source_ref}


def _reference(
    reference_id: str,
    path: str,
    lines: str,
    revision: str = STAGE_A_PARENT_COMMIT,
) -> dict[str, str]:
    return {
        "lines": lines,
        "path": path,
        "reference_id": reference_id,
        "revision": revision,
    }


def _with_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["manifest_sha256"] = hash_json(result)
    return result


def _generation_references() -> list[dict[str, str]]:
    hv = REVISIONS["hunyuan_video"]
    return [
        _reference(
            "PLAN-GENERATION",
            "docs/hunyuan_video_clean_rebuild_research_plan_zh.md",
            "68-111",
            "worktree",
        ),
        _reference(
            "HV-CONFIG-MODEL",
            "reference/hunyuan_video/code/hyvideo/config.py",
            "22-45,59-170,219-269",
            hv,
        ),
        _reference(
            "HV-CONFIG-GENERATION",
            "reference/hunyuan_video/code/hyvideo/config.py",
            "295-347",
            hv,
        ),
        _reference(
            "HV-CONFIG-SAMPLER",
            "reference/hunyuan_video/code/hyvideo/config.py",
            "175-215",
            hv,
        ),
        _reference(
            "HV-OFFICIAL-COMMAND",
            "reference/hunyuan_video/code/README.md",
            "314-360",
            hv,
        ),
        _reference(
            "HV-PREDICT-SAMPLER",
            "reference/hunyuan_video/code/hyvideo/inference.py",
            "580-616",
            hv,
        ),
        _reference(
            "HV-CFG-SEMANTICS",
            "reference/hunyuan_video/code/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py",
            "642-648",
            hv,
        ),
        _reference(
            "HV-ARCHITECTURE",
            "reference/hunyuan_video/code/hyvideo/modules/models.py",
            "744-753",
            hv,
        ),
    ]


def _base_model() -> dict[str, Any]:
    return {
        "checkpoint": evidence(
            "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            "project_frozen_and_source_code",
            "PLAN-GENERATION;HV-CONFIG-MODEL",
        ),
        "load_key": evidence(
            "module", "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-MODEL"
        ),
        "model": evidence(
            "HYVideo-T/2-cfgdistill",
            "project_frozen_and_source_code",
            "PLAN-GENERATION;HV-CONFIG-MODEL",
        ),
        "source_commit": evidence(
            REVISIONS["hunyuan_video"], "pinned_commit", "PLAN-GENERATION"
        ),
        "transformer_double_stream_blocks": evidence(
            20, "project_frozen_and_source_code", "PLAN-GENERATION;HV-ARCHITECTURE"
        ),
        "transformer_single_stream_blocks": evidence(
            40, "project_frozen_and_source_code", "PLAN-GENERATION;HV-ARCHITECTURE"
        ),
    }


def _precision() -> dict[str, Any]:
    return {
        "autocast": evidence(True, "project_frozen", "PLAN-GENERATION"),
        "dit": evidence(
            "bf16", "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-MODEL"
        ),
        "text_encoder_1": evidence(
            "fp16", "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-MODEL"
        ),
        "text_encoder_2": evidence(
            "fp16", "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-MODEL"
        ),
        "vae": evidence(
            "fp16", "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-MODEL"
        ),
    }


def _sampler() -> dict[str, Any]:
    return {
        "cfg_enabled": evidence(
            False, "derived_from_source_code", "HV-CFG-SEMANTICS"
        ),
        "embedded_guidance_scale": evidence(
            6.0, "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-GENERATION"
        ),
        "flow_reverse": evidence(
            True, "project_frozen_and_official_command", "PLAN-GENERATION;HV-OFFICIAL-COMMAND"
        ),
        "flow_shift": evidence(
            7.0, "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-SAMPLER"
        ),
        "guidance_scale": evidence(
            1.0, "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-GENERATION"
        ),
        "scheduler": evidence(
            "FlowMatchDiscreteScheduler", "project_frozen_and_source_code", "PLAN-GENERATION;HV-PREDICT-SAMPLER"
        ),
        "solver": evidence(
            "euler", "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-SAMPLER"
        ),
    }


def _prompt_processing() -> dict[str, Any]:
    return {
        "apply_final_norm": evidence(False, "source_code_default", "HV-CONFIG-MODEL"),
        "hidden_state_skip_layer": evidence(2, "source_code_default", "HV-CONFIG-MODEL"),
        "prompt_rewrite_enabled": evidence(False, "project_frozen", "PLAN-GENERATION"),
        "prompt_template": evidence(
            "dit-llm-encode-video", "source_code_default", "HV-CONFIG-MODEL"
        ),
        "text_encoder_1": evidence("llm", "source_code_default", "HV-CONFIG-MODEL"),
        "text_encoder_2": evidence("clipL", "source_code_default", "HV-CONFIG-MODEL"),
    }


def _generation_protocol(
    protocol_id: str,
    *,
    height: int,
    width: int,
    frames: int,
    cpu_offload: bool,
    role: str,
    timing_label: str,
    gpu_model: str | None,
    independent_shards_per_node: int,
) -> dict[str, Any]:
    return {
        "generation": {
            "batch_size": evidence(1, "project_frozen", "PLAN-GENERATION"),
            "fps": evidence(24, "project_frozen", "PLAN-GENERATION"),
            "frames": evidence(frames, "project_frozen", "PLAN-GENERATION"),
            "height": evidence(height, "project_frozen", "PLAN-GENERATION"),
            "num_videos_per_prompt": evidence(1, "project_frozen", "PLAN-GENERATION"),
            "steps": evidence(50, "project_frozen_and_source_code", "PLAN-GENERATION;HV-CONFIG-GENERATION"),
            "width": evidence(width, "project_frozen", "PLAN-GENERATION"),
        },
        "model_identity": _base_model(),
        "precision": _precision(),
        "prompt_processing": _prompt_processing(),
        "protocol_id": protocol_id,
        "provenance_refs": [
            "PLAN-GENERATION",
            "HV-CONFIG-MODEL",
            "HV-CONFIG-GENERATION",
            "HV-CONFIG-SAMPLER",
            "HV-CFG-SEMANTICS",
        ],
        "role": role,
        "runtime_topology": {
            "cpu_offload": evidence(cpu_offload, "project_frozen", "PLAN-GENERATION"),
            "gpu_model": evidence(gpu_model, "project_frozen", "PLAN-GENERATION"),
            "gpus_per_video": evidence(1, "project_frozen", "PLAN-GENERATION"),
            "independent_shards_per_node": evidence(
                independent_shards_per_node, "project_frozen", "PLAN-GENERATION"
            ),
            "sequence_parallel": evidence(False, "project_frozen", "PLAN-GENERATION"),
            "timing_label": evidence(timing_label, "project_frozen", "PLAN-GENERATION"),
        },
        "sampler": _sampler(),
    }


def build_generation_manifest() -> dict[str, Any]:
    manifest = {
        "canonical_json": "UTF-8; sorted keys; no insignificant whitespace; trailing LF excluded from hash",
        "protocols": [
            _generation_protocol(
                "HY-Official-720",
                height=720,
                width=1280,
                frames=129,
                cpu_offload=True,
                role="official identity, parity, and final-winner resolution transfer",
                timing_label="official parity protocol",
                gpu_model=None,
                independent_shards_per_node=1,
            ),
            _generation_protocol(
                "HY-CachePaper-480",
                height=480,
                width=640,
                frames=65,
                cpu_offload=False,
                role="unified baseline and Golden Path protocol",
                timing_label="unified fair protocol; not author paper-compatible timing",
                gpu_model="NVIDIA H100",
                independent_shards_per_node=4,
            ),
        ],
        "references": _generation_references(),
        "schema": "hunyuan_video.generation_protocols.v1",
        "source_revisions": {
            "stage_a_parent": STAGE_A_PARENT_COMMIT,
            "tencent_hunyuan_video": REVISIONS["hunyuan_video"],
        },
    }
    return _with_hash(manifest)


def build_source_generation_manifest() -> dict[str, Any]:
    """Build executable source-lane identities without changing unified hashes."""

    manifest = {
        "canonical_json": (
            "UTF-8; sorted keys; no insignificant whitespace; trailing LF excluded from hash"
        ),
        "protocols": [
            _generation_protocol(
                "HY-TaylorSource-480",
                height=480,
                width=640,
                frames=65,
                cpu_offload=True,
                role=(
                    "project-executable TaylorSeer source lane; source compatibility "
                    "is limited to fields disclosed by SRC-TAYLOR-480-O1"
                ),
                timing_label=(
                    "TaylorSeer released 480p execution topology; paper H20 timing "
                    "remains a separate claim"
                ),
                gpu_model="NVIDIA H100",
                independent_shards_per_node=1,
            )
        ],
        "references": _generation_references(),
        "schema": "hunyuan_video.source_generation_protocols.v1",
        "source_protocol_registry_id": "SRC-TAYLOR-480-O1",
        "source_revisions": {
            "stage_a_parent": STAGE_A_PARENT_COMMIT,
            "taylorseer": REVISIONS["taylorseer"],
            "tencent_hunyuan_video": REVISIONS["hunyuan_video"],
        },
    }
    return _with_hash(manifest)


def _source_references() -> list[dict[str, str]]:
    return [
        _reference(
            "PLAN-SOURCE-LABELS",
            "docs/hunyuan_video_clean_rebuild_research_plan_zh.md",
            "240-278",
            "worktree",
        ),
        _reference(
            "TEA-README-OPS",
            "reference/teacache/code/TeaCache4HunyuanVideo/README.md",
            "4-35",
            REVISIONS["teacache"],
        ),
        _reference(
            "TEA-CODE-GATE",
            "reference/teacache/code/TeaCache4HunyuanVideo/teacache_sample_video.py",
            "82-119,218-226",
            REVISIONS["teacache"],
        ),
        _reference(
            "SEA-PAPER-PROTOCOL",
            "reference/seacache/paper/sec/4_experiment.tex",
            "60-85",
        ),
        _reference(
            "SEA-PAPER-ROWS",
            "reference/seacache/paper/sec/4_experiment.tex",
            "102-121",
        ),
        _reference(
            "SEA-CODE-README",
            "reference/seacache/code/HunyuanVideo/README.md",
            "1-50",
            REVISIONS["seacache"],
        ),
        _reference(
            "SEA-CODE-GATE",
            "reference/seacache/code/HunyuanVideo/seacache_generate.py",
            "80-127,227-252",
            REVISIONS["seacache"],
        ),
        _reference(
            "TAYLOR-PAPER-PROTOCOL",
            "reference/taylorseer/paper/sec/4_experiments.tex",
            "7-19,55-60",
        ),
        _reference(
            "TAYLOR-PAPER-ROWS",
            "reference/taylorseer/paper/tab/HunyuanVideo-Metrics.tex",
            "19-47",
        ),
        _reference(
            "TAYLOR-CODE-VBENCH",
            "reference/taylorseer/code/TaylorSeer-HunyuanVideo/eval/sample_vbench.sh",
            "53-64",
            REVISIONS["taylorseer"],
        ),
        _reference(
            "TAYLOR-CODE-CONFIG",
            "reference/taylorseer/code/TaylorSeer-HunyuanVideo/hyvideo/modules/cache_functions/cache_init.py",
            "32-59,103-120",
            REVISIONS["taylorseer"],
        ),
        _reference(
            "TAYLOR-CODE-DOUBLE-SLOTS",
            "reference/taylorseer/code/TaylorSeer-HunyuanVideo/hyvideo/modules/models.py",
            "235-308",
            REVISIONS["taylorseer"],
        ),
        _reference(
            "TAYLOR-CODE-SINGLE-SLOTS",
            "reference/taylorseer/code/TaylorSeer-HunyuanVideo/hyvideo/modules/models.py",
            "481-554",
            REVISIONS["taylorseer"],
        ),
        _reference(
            "HICACHE-PAPER-ROWS",
            "reference/hicache/paper/hicache_arxiv.tex",
            "337-363,447-447",
        ),
        _reference(
            "HICACHE-CODE-NO-BACKEND",
            "reference/hicache/code/README.md",
            "141-147,180-190",
            REVISIONS["hicache"],
        ),
    ]


def _common_unknown_execution(source_ref: str) -> dict[str, Any]:
    return {
        "checkpoint": unknown(source_ref),
        "checkpoint_digest": unknown(source_ref),
        "embedded_guidance_scale": unknown(source_ref),
        "guidance_scale": unknown(source_ref),
        "seed_values": unknown(source_ref),
    }


def _op(
    operating_point_id: str,
    classification: str,
    parameters: dict[str, Any],
    *,
    unknown_fields: list[str] | None = None,
    blocking_unknown_fields: list[str] | None = None,
    deviations: list[dict[str, str]] | None = None,
    reported_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "blocking_unknown_fields": blocking_unknown_fields or [],
        "classification": classification,
        "deviations": deviations or [],
        "operating_point_id": operating_point_id,
        "parameters": parameters,
        "unknown_fields": unknown_fields or [],
    }
    if reported_timing is not None:
        result["reported_timing"] = reported_timing
    return result


def _tea_540() -> dict[str, Any]:
    execution = _common_unknown_execution("TEA-README-OPS")
    execution.update(
        {
            "cpu_offload": unknown("TEA-README-OPS"),
            "fps": evidence(24, "released_code_save_default", "TEA-CODE-GATE"),
            "frames": unknown("TEA-README-OPS"),
            "height": unknown("TEA-README-OPS"),
            "resolution_label": evidence("540p", "released_readme_claim", "TEA-README-OPS"),
            "scheduler": unknown("TEA-README-OPS"),
            "steps": unknown("TEA-README-OPS"),
            "width": unknown("TEA-README-OPS"),
        }
    )
    blocking = [
        "external_hunyuan_commit",
        "frames",
        "height",
        "scheduler",
        "seed_values",
        "steps",
        "width",
    ]
    return {
        "allowed_label": "TeaCache released-code 540p claim",
        "blocking_unknown_fields": blocking,
        "classification": "claim_only",
        "deviations": [
            deviation(
                "documentation_only",
                "The README reports 540p latency but provides no 540p command or complete executable protocol.",
                "TEA-README-OPS",
            )
        ],
        "execution": execution,
        "method": "TeaCache",
        "operating_points": [
            _op(
                "OP-TEA-540-D010",
                "claim_only",
                {"rel_l1_threshold": evidence(0.10, "released_readme_claim", "TEA-README-OPS")},
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
            ),
            _op(
                "OP-TEA-540-D015",
                "claim_only",
                {"rel_l1_threshold": evidence(0.15, "released_readme_claim", "TEA-README-OPS")},
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
            ),
        ],
        "provenance_refs": ["PLAN-SOURCE-LABELS", "TEA-README-OPS", "TEA-CODE-GATE"],
        "source_protocol_id": "SRC-TEA-540",
        "unknown_fields": blocking,
    }


def _tea_720() -> dict[str, Any]:
    execution = _common_unknown_execution("TEA-README-OPS")
    execution.update(
        {
            "cpu_offload": evidence(True, "released_readme_command", "TEA-README-OPS"),
            "fps": evidence(24, "released_code_save_default", "TEA-CODE-GATE"),
            "frames": evidence(129, "released_readme_command", "TEA-README-OPS"),
            "height": evidence(720, "released_readme_command", "TEA-README-OPS"),
            "scheduler": unknown("TEA-README-OPS"),
            "steps": evidence(50, "released_readme_command", "TEA-README-OPS"),
            "width": evidence(1280, "released_readme_command", "TEA-README-OPS"),
        }
    )
    blocking = ["external_hunyuan_commit", "scheduler", "seed_values"]
    common_deviations = [
        deviation(
            "external_base_unpinned",
            "The patch script imports an external HunyuanVideo checkout without pinning its commit.",
            "TEA-README-OPS",
        ),
        deviation(
            "hardcoded_interface",
            "The script has no threshold CLI and hardcodes rel_l1_thresh=0.15.",
            "TEA-CODE-GATE",
        ),
    ]
    return {
        "allowed_label": "TeaCache released-code",
        "blocking_unknown_fields": blocking,
        "classification": "conditional",
        "deviations": common_deviations,
        "execution": execution,
        "method": "TeaCache",
        "operating_points": [
            _op(
                "OP-TEA-720-D010",
                "conditional",
                {"rel_l1_threshold": evidence(0.10, "released_readme_and_code_comment", "TEA-README-OPS;TEA-CODE-GATE")},
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
                deviations=common_deviations
                + [
                    deviation(
                        "source_edit_required",
                        "Change the hardcoded 0.15 assignment to 0.10 before execution.",
                        "TEA-CODE-GATE",
                    )
                ],
            ),
            _op(
                "OP-TEA-720-D015",
                "conditional",
                {"rel_l1_threshold": evidence(0.15, "released_code", "TEA-CODE-GATE")},
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
                deviations=common_deviations,
            ),
        ],
        "provenance_refs": ["PLAN-SOURCE-LABELS", "TEA-README-OPS", "TEA-CODE-GATE"],
        "source_protocol_id": "SRC-TEA-720",
        "unknown_fields": blocking,
    }


def _sea_paper_family(
    source_protocol_id: str,
    method: str,
    allowed_label: str,
    points: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    execution = _common_unknown_execution("SEA-PAPER-PROTOCOL")
    execution.update(
        {
            "cpu_offload": unknown("SEA-PAPER-PROTOCOL"),
            "frames": evidence(65, "paper", "SEA-PAPER-PROTOCOL"),
            "height": unknown("SEA-PAPER-PROTOCOL"),
            "initial_full_step_indices": evidence([0, 1, 2], "paper", "SEA-PAPER-PROTOCOL"),
            "resolution_label": evidence("480p", "paper", "SEA-PAPER-PROTOCOL"),
            "scheduler": unknown("SEA-PAPER-PROTOCOL"),
            "steps": evidence(50, "paper", "SEA-PAPER-PROTOCOL"),
            "width": unknown("SEA-PAPER-PROTOCOL"),
        }
    )
    blocking = [
        "base_model_commit",
        "cache_implementation_commit",
        "checkpoint",
        "cpu_offload",
        "embedded_guidance_scale",
        "guidance_scale",
        "height",
        "scheduler",
        "seed_values",
        "width",
    ]
    family_deviations = [
        deviation(
            "paper_only_protocol",
            "The paper discloses 480p/65f/50-step rows and first-three refresh but not a complete executable identity.",
            "SEA-PAPER-PROTOCOL;SEA-PAPER-ROWS",
        )
    ]
    return {
        "allowed_label": allowed_label,
        "blocking_unknown_fields": blocking,
        "classification": "claim_only",
        "deviations": family_deviations,
        "execution": execution,
        "method": method,
        "operating_points": [
            _op(
                point_id,
                "claim_only",
                parameters,
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
                deviations=family_deviations,
            )
            for point_id, parameters in points
        ],
        "provenance_refs": ["PLAN-SOURCE-LABELS", "SEA-PAPER-PROTOCOL", "SEA-PAPER-ROWS"],
        "source_protocol_id": source_protocol_id,
        "unknown_fields": blocking,
    }


def _sea_code() -> dict[str, Any]:
    execution = _common_unknown_execution("SEA-CODE-README")
    execution.update(
        {
            "cpu_offload": evidence(True, "released_readme_command", "SEA-CODE-README"),
            "final_full_step_index": evidence(49, "released_code", "SEA-CODE-GATE"),
            "first_full_step_index": evidence(0, "released_code", "SEA-CODE-GATE"),
            "frames": evidence(33, "released_readme_command", "SEA-CODE-README"),
            "height": evidence(720, "released_readme_command", "SEA-CODE-README"),
            "scheduler": unknown("SEA-CODE-README"),
            "steps": evidence(50, "released_readme_command", "SEA-CODE-README"),
            "threshold_cli_effective": evidence(False, "released_code_audit", "SEA-CODE-README;SEA-CODE-GATE"),
            "width": evidence(1280, "released_readme_command", "SEA-CODE-README"),
        }
    )
    blocking = ["external_hunyuan_commit", "scheduler", "seed_values"]
    deviations = [
        deviation(
            "external_base_unpinned",
            "The patch script imports an external HunyuanVideo checkout without pinning its commit.",
            "SEA-CODE-README",
        ),
        deviation(
            "broken_threshold_interface",
            "The README passes --rel_l1_thresh, but the patch defines no parser option and hardcodes 0.20.",
            "SEA-CODE-README;SEA-CODE-GATE",
        ),
        deviation(
            "paper_schedule_mismatch",
            "Released code forces only first/last full, not the paper's first-three video refresh.",
            "SEA-PAPER-PROTOCOL;SEA-CODE-GATE",
        ),
    ]
    return {
        "allowed_label": "SeaCache released-code parity",
        "blocking_unknown_fields": blocking,
        "classification": "conditional",
        "deviations": deviations,
        "execution": execution,
        "method": "SeaCache",
        "operating_points": [
            _op(
                "OP-SEA-720-CODE-33F-D020",
                "conditional",
                {"rel_l1_threshold": evidence(0.20, "released_code_hardcoded", "SEA-CODE-GATE")},
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
                deviations=deviations,
            )
        ],
        "provenance_refs": [
            "PLAN-SOURCE-LABELS",
            "SEA-CODE-README",
            "SEA-CODE-GATE",
            "SEA-PAPER-PROTOCOL",
        ],
        "source_protocol_id": "SRC-SEA-720-CODE-33F",
        "unknown_fields": blocking,
    }


def _taylor() -> dict[str, Any]:
    execution = {
        "applied_prediction_outputs": evidence(
            120,
            "source_code_audit_20x4_plus_40x1",
            "TAYLOR-CODE-DOUBLE-SLOTS;TAYLOR-CODE-SINGLE-SLOTS",
        ),
        "checkpoint": evidence(
            "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            "source_code_default",
            "TAYLOR-CODE-VBENCH",
        ),
        "checkpoint_digest": unknown("TAYLOR-CODE-VBENCH"),
        "cpu_offload": evidence(True, "released_script", "TAYLOR-CODE-VBENCH"),
        "embedded_guidance_scale": evidence(6.0, "source_code_default", "TAYLOR-CODE-VBENCH"),
        "frames": evidence(65, "released_script", "TAYLOR-CODE-VBENCH"),
        "guidance_scale": evidence(1.0, "source_code_default", "TAYLOR-CODE-VBENCH"),
        "height": evidence(480, "released_script", "TAYLOR-CODE-VBENCH"),
        "history_slots": evidence(
            160,
            "source_code_audit_20x4_plus_40x2",
            "TAYLOR-CODE-CONFIG;TAYLOR-CODE-DOUBLE-SLOTS;TAYLOR-CODE-SINGLE-SLOTS",
        ),
        "seed_values": evidence([42, 43, 44, 45, 46], "project_frozen_source_lane", "PLAN-SOURCE-LABELS"),
        "steps": evidence(50, "released_script_and_paper", "TAYLOR-CODE-VBENCH;TAYLOR-PAPER-PROTOCOL"),
        "width": evidence(640, "released_script", "TAYLOR-CODE-VBENCH"),
    }
    latencies = {3: 113.44, 5: 68.42, 6: 61.99}
    speedups = {3: 2.81, 5: 4.65, 6: 5.13}
    points = []
    for interval in (3, 5, 6):
        deviations = []
        if interval != 5:
            deviations.append(
                deviation(
                    "source_parameter_edit_required",
                    f"Set the hardcoded fresh_threshold from 5 to {interval}; the backend remains source-executable.",
                    "TAYLOR-CODE-CONFIG",
                )
            )
        points.append(
            _op(
                f"OP-TAYLOR-480-O1-N{interval}",
                "executable",
                {
                    "activation_interval": evidence(interval, "paper_row", "TAYLOR-PAPER-ROWS"),
                    "taylor_order": evidence(1, "paper_row_and_source_code", "TAYLOR-PAPER-ROWS;TAYLOR-CODE-CONFIG"),
                },
                unknown_fields=["checkpoint_digest", "upstream_tencent_base_commit"],
                deviations=deviations,
                reported_timing={
                    "hardware": evidence("NVIDIA H20 96GB", "paper", "TAYLOR-PAPER-PROTOCOL"),
                    "lane": evidence("paper timing; separate from released source execution", "audit_classification", "TAYLOR-PAPER-PROTOCOL;TAYLOR-CODE-VBENCH"),
                    "latency_seconds": evidence(latencies[interval], "paper_row", "TAYLOR-PAPER-ROWS"),
                    "speedup": evidence(speedups[interval], "paper_row", "TAYLOR-PAPER-ROWS"),
                },
            )
        )
    return {
        "allowed_label": "TaylorSeer paper/source lane",
        "blocking_unknown_fields": [],
        "classification": "executable",
        "deviations": [
            deviation(
                "timing_provenance_separate",
                "Paper latency is an H20 claim and is not attributed to the released 480p execution script.",
                "TAYLOR-PAPER-PROTOCOL;TAYLOR-CODE-VBENCH",
            )
        ],
        "execution": execution,
        "method": "TaylorSeer",
        "operating_points": points,
        "provenance_refs": [
            "PLAN-SOURCE-LABELS",
            "TAYLOR-PAPER-PROTOCOL",
            "TAYLOR-PAPER-ROWS",
            "TAYLOR-CODE-VBENCH",
            "TAYLOR-CODE-CONFIG",
            "TAYLOR-CODE-DOUBLE-SLOTS",
            "TAYLOR-CODE-SINGLE-SLOTS",
        ],
        "source_protocol_id": "SRC-TAYLOR-480-O1",
        "unknown_fields": ["checkpoint_digest", "upstream_tencent_base_commit"],
    }


def _hicache() -> dict[str, Any]:
    blocking = [
        "base_model_commit",
        "cache_site",
        "checkpoint",
        "frames",
        "height",
        "history_topology",
        "scheduler",
        "seed_values",
        "sigma",
        "width",
    ]
    execution = {field: unknown("HICACHE-PAPER-ROWS;HICACHE-CODE-NO-BACKEND") for field in blocking}
    deviations = [
        deviation(
            "no_released_hunyuan_backend",
            "The pinned code has no HunyuanVideo implementation; executable project work must be labeled an audited adaptation.",
            "HICACHE-CODE-NO-BACKEND",
        )
    ]
    points = []
    for interval, order in ((6, 1), (7, 1), (7, 2)):
        points.append(
            _op(
                f"OP-HICACHE-HY-N{interval}-O{order}",
                "claim_only",
                {
                    "activation_interval": evidence(interval, "paper_claim", "HICACHE-PAPER-ROWS"),
                    "hermite_order": evidence(order, "paper_claim", "HICACHE-PAPER-ROWS"),
                    "sigma": unknown("HICACHE-PAPER-ROWS"),
                },
                unknown_fields=blocking,
                blocking_unknown_fields=blocking,
                deviations=deviations,
            )
        )
    return {
        "allowed_label": "HiCache Hunyuan audited adaptation",
        "blocking_unknown_fields": blocking,
        "classification": "claim_only",
        "deviations": deviations,
        "execution": execution,
        "method": "HiCache",
        "operating_points": points,
        "provenance_refs": ["PLAN-SOURCE-LABELS", "HICACHE-PAPER-ROWS", "HICACHE-CODE-NO-BACKEND"],
        "source_protocol_id": "SRC-HICACHE-HY-CLAIM",
        "unknown_fields": blocking,
    }


def build_source_manifest() -> dict[str, Any]:
    sea_paper = _sea_paper_family(
        "SRC-SEA-480-PAPER",
        "SeaCache",
        "SeaCache paper reconstruction",
        [
            ("OP-SEA-480-PAPER-D019", {"rel_l1_threshold": evidence(0.19, "paper_row", "SEA-PAPER-ROWS")}),
            ("OP-SEA-480-PAPER-D035", {"rel_l1_threshold": evidence(0.35, "paper_row", "SEA-PAPER-ROWS")}),
        ],
    )
    sea_tea = _sea_paper_family(
        "SRC-SEA-TEA",
        "TeaCache",
        "SeaCache-paper TeaCache comparator",
        [
            ("OP-SEA-TEA-D012", {"rel_l1_threshold": evidence(0.12, "paper_row", "SEA-PAPER-ROWS")}),
            ("OP-SEA-TEA-D020", {"rel_l1_threshold": evidence(0.20, "paper_row", "SEA-PAPER-ROWS")}),
        ],
    )
    sea_taylor = _sea_paper_family(
        "SRC-SEA-TAYLOR-O2",
        "TaylorSeer",
        "SeaCache-paper Taylor comparator",
        [
            (
                "OP-SEA-TAYLOR-O2-S2",
                {
                    "activation_stride": evidence(2, "paper_row", "SEA-PAPER-ROWS"),
                    "taylor_order": evidence(2, "paper_protocol", "SEA-PAPER-PROTOCOL"),
                },
            ),
            (
                "OP-SEA-TAYLOR-O2-S3",
                {
                    "activation_stride": evidence(3, "paper_row", "SEA-PAPER-ROWS"),
                    "taylor_order": evidence(2, "paper_protocol", "SEA-PAPER-PROTOCOL"),
                },
            ),
        ],
    )
    protocols = [
        _tea_540(),
        _tea_720(),
        sea_paper,
        _sea_code(),
        _taylor(),
        sea_tea,
        sea_taylor,
        _hicache(),
    ]
    for protocol in protocols:
        execution_nulls = {
            field
            for field, item in protocol.get("execution", {}).items()
            if isinstance(item, dict)
            and item.get("evidence_status") == "unknown"
            and item.get("value") is None
        }
        protocol["unknown_fields"] = sorted(
            set(protocol.get("unknown_fields", [])) | execution_nulls
        )
        fields = sorted(
            set(protocol.get("unknown_fields", []))
            | set(protocol.get("blocking_unknown_fields", []))
        )
        protocol["unknown_values"] = {
            field: unknown(f"{protocol['source_protocol_id']}:undisclosed")
            for field in fields
        }
        for point in protocol["operating_points"]:
            point_fields = sorted(
                set(point.get("unknown_fields", []))
                | set(point.get("blocking_unknown_fields", []))
            )
            point["unknown_values"] = {
                field: unknown(f"{point['operating_point_id']}:undisclosed")
                for field in point_fields
            }
    manifest = {
        "canonical_json": "UTF-8; sorted keys; no insignificant whitespace; trailing LF excluded from hash",
        "classification_definitions": {
            "claim_only": "Published or documented claim lacks a complete source-executable protocol.",
            "conditional": "Released code exists but requires an unresolved external dependency or source/interface repair.",
            "executable": "Pinned source contains the backend and the row can be configured without reconstructing missing method logic.",
        },
        "protocols": protocols,
        "references": _source_references(),
        "schema": "hunyuan_video.source_protocols.v1",
        "source_revisions": {
            "hicache": REVISIONS["hicache"],
            "stage_a_parent": STAGE_A_PARENT_COMMIT,
            "seacache": REVISIONS["seacache"],
            "taylorseer": REVISIONS["taylorseer"],
            "teacache": REVISIONS["teacache"],
            "tencent_hunyuan_video": REVISIONS["hunyuan_video"],
        },
    }
    return _with_hash(manifest)


def write_manifests(
    generation_output: Path = ROOT / GENERATION_OUTPUT_PATH,
    source_generation_output: Path = ROOT / SOURCE_GENERATION_OUTPUT_PATH,
    source_output: Path = ROOT / SOURCE_OUTPUT_PATH,
) -> None:
    generation_output.write_bytes(canonical_json_bytes(build_generation_manifest()) + b"\n")
    source_generation_output.write_bytes(
        canonical_json_bytes(build_source_generation_manifest()) + b"\n"
    )
    source_output.write_bytes(canonical_json_bytes(build_source_manifest()) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-output", type=Path, default=ROOT / GENERATION_OUTPUT_PATH)
    parser.add_argument(
        "--source-generation-output",
        type=Path,
        default=ROOT / SOURCE_GENERATION_OUTPUT_PATH,
    )
    parser.add_argument("--source-output", type=Path, default=ROOT / SOURCE_OUTPUT_PATH)
    args = parser.parse_args()
    write_manifests(
        args.generation_output,
        args.source_generation_output,
        args.source_output,
    )


if __name__ == "__main__":
    main()
