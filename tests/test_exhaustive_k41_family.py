from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from analysis.summarize_exhaustive_k41_family import (
    PRIMARY_RANKS,
    load_manifest,
    main as summarize_main,
    summarize_environment,
)
from flux.exhaustive_k41_family_runner import (
    DEFAULT_MANIFEST,
    DEFAULT_SAVED_RANKS,
    DEFAULT_SCHEDULE_DIR,
    EXPECTED_SAVED_FULL_STEPS,
    PAIR_SCHEMA,
    _image_paths,
    load_candidates,
    validate_completed_pair,
)


@pytest.fixture
def scored_manifest(tmp_path: Path) -> Path:
    """Use packaged candidate definitions with artificial discovery scores."""
    with DEFAULT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    path = tmp_path / "synthetic_scored_manifest.tsv"
    fields = [*rows[0], "mean_psnr_db", "min_psnr_db"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows):
            mean = 40.0 - index / 100.0
            writer.writerow({**row, "mean_psnr_db": mean, "min_psnr_db": mean - 0.5})
    return path


def test_frozen_candidate_family_and_saved_paths() -> None:
    candidates = load_candidates(DEFAULT_MANIFEST, DEFAULT_SCHEDULE_DIR)
    assert len(candidates) == 337
    assert len({candidate.rank for candidate in candidates}) == 337
    assert len({candidate.bits for candidate in candidates}) == 337

    saved = dict(DEFAULT_SAVED_RANKS)
    by_rank = {candidate.rank: candidate for candidate in candidates}
    assert {
        label: by_rank[rank].full_steps for label, rank in saved.items()
    } == EXPECTED_SAVED_FULL_STEPS


def test_completed_pair_requires_protocol_rows_and_saved_images(tmp_path: Path) -> None:
    saved = dict(DEFAULT_SAVED_RANKS)
    ranks = [1, 2]
    protocol = {"candidate_count": 2, "base_seed": 41}
    images = _image_paths(tmp_path, 7, saved)
    for path in images.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    pair_path = tmp_path / "pairs" / "pair_00007.json"
    pair_path.parent.mkdir(parents=True)
    pair_path.write_text(
        json.dumps(
            {
                "schema": PAIR_SCHEMA,
                "protocol": protocol,
                "prompt_idx": 7,
                "seed": 48,
                "candidates": [{"rank": 1}, {"rank": 2}],
            }
        ),
        encoding="utf-8",
    )
    validate_completed_pair(
        pair_path,
        prompt_idx=7,
        seed=48,
        protocol=protocol,
        candidate_ranks=ranks,
        image_paths=images,
    )


def test_family_summary_recovers_synthetic_discovery_order(scored_manifest: Path) -> None:
    manifest = load_manifest(scored_manifest)
    ranks = list(manifest)
    pairs = []
    for prompt_idx, offset in ((10, 0.0), (11, 0.1), (12, -0.1)):
        pairs.append(
            {
                "prompt_idx": prompt_idx,
                "seed": 41 + prompt_idx,
                "candidates": [
                    {
                        "rank": rank,
                        "psnr_db": manifest[rank]["discovery_mean_psnr_db"] + offset,
                    }
                    for rank in ranks
                ],
            }
        )

    summary, candidate_rows, primary_rows, frequency_rows = summarize_environment(
        "synthetic", pairs, manifest
    )
    assert summary["n_pairs"] == 3
    assert abs(summary["spearman_discovery_vs_heldout"] - 1.0) < 1e-12
    assert summary["discovery_top64_retained"] == 64
    assert len(candidate_rows) == 337
    assert {row["rank"] for row in primary_rows} == {
        PRIMARY_RANKS["mean_optimal"],
        PRIMARY_RANKS["robust_optimal"],
    }
    assert len(frequency_rows) == 50


def test_summary_cli_writes_complete_scientific_outputs(
    tmp_path: Path, monkeypatch, scored_manifest: Path
) -> None:
    manifest = load_manifest(scored_manifest)
    ranks = list(manifest)
    environment = tmp_path / "drawbench_s41"
    pairs_dir = environment / "pairs"
    pairs_dir.mkdir(parents=True)
    protocol = {"model_commit": "frozen", "base_seed": 41}
    for prompt_idx, offset in ((0, 0.0), (1, 0.2)):
        (pairs_dir / f"pair_{prompt_idx:05d}.json").write_text(
            json.dumps(
                {
                    "schema": PAIR_SCHEMA,
                    "protocol": protocol,
                    "prompt_idx": prompt_idx,
                    "seed": 41 + prompt_idx,
                    "candidates": [
                        {
                            "rank": rank,
                            "psnr_db": manifest[rank]["discovery_mean_psnr_db"]
                            + offset,
                        }
                        for rank in ranks
                    ],
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_exhaustive_k41_family.py",
            "--environment",
            f"drawbench_s41={environment}",
            "--expected_pairs",
            "drawbench_s41=2",
            "--manifest",
            str(scored_manifest),
            "--output_dir",
            str(output),
        ],
    )
    assert summarize_main() == 0
    assert (output / "candidate_statistics.tsv").is_file()
    assert (output / "primary_vs_budcache.tsv").is_file()
    assert (output / "top64_full_step_frequency.tsv").is_file()
    assert (output / "equal_environment_macro.tsv").is_file()
    summary = json.loads(
        (output / "family_migration_summary.json").read_text(encoding="utf-8")
    )
    assert summary["environment_count"] == 1
    assert summary["candidate_count"] == 337
