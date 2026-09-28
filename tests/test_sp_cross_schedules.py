from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis import build_sp_cross_schedules as builder
from flux.sp_cross_runner import load_schedule_file as flux_load_schedule_file
from qwen_image.sp_cross_runner import load_schedule_file as qwen_load_schedule_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_DIR = REPO_ROOT / "resources" / "sp_cross_schedules"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_check_schedule_accepts_a_well_formed_bitstring() -> None:
    bits = "0" * 3 + "1" * 2
    assert builder.check_schedule(bits, num_steps=5, cache_count=2, label="t") == 2


@pytest.mark.parametrize(
    ("bits", "cache_count", "match"),
    [
        ("0011", 2, "characters of 0/1"),
        ("0012 ", 2, "characters of 0/1"),
        ("00011", 3, "disagrees with source cache_count"),
        ("10011", 3, "step 0 must be a full step"),
    ],
)
def test_check_schedule_rejects_bad_rows(bits: str, cache_count: int, match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        builder.check_schedule(bits, num_steps=5, cache_count=cache_count, label="t")


@pytest.fixture
def schedule_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Small generated inputs test construction without measured path counts."""
    fixed, native = [], []
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            bits = "0" + "1" * k + "0" * (49 - k)
            for method in builder.FIXED_SCHEDULES.values():
                fixed.append({"model": model, "target_k": k, "method": method,
                              "cache_count": k, "schedule": bits})
            for method in builder.NATIVE_SCHEDULES.values():
                base = {"model": model, "target_k": k, "method": method,
                        "dataset": "synthetic_discovery"}
                native.append({**base, "cache_count": k, "schedule": bits,
                               "count": 20, "mass": 0.2,
                               "rank": 2 if method == "sencache" else 1})
                if method == "sencache":
                    native.append({**base, "cache_count": k + 1,
                                   "schedule": "0" + "1" * (k + 1) + "0" * (48 - k),
                                   "count": 80, "mass": 0.8, "rank": 1})
    paths = tmp_path / "fixed.tsv", tmp_path / "counts.tsv"
    builder.write_tsv(paths[0], fixed, list(fixed[0]))
    builder.write_tsv(paths[1], native, list(native[0]))
    return paths


def test_builder_emits_one_file_per_model_budget_schedule(
    tmp_path: Path, schedule_sources: tuple[Path, Path]
) -> None:
    fixed_path, counts_path = schedule_sources
    argv = [
        "--out_dir",
        str(tmp_path),
        "--fixed_paths",
        str(fixed_path),
        "--path_counts",
        str(counts_path),
        "--dataset",
        "synthetic_discovery",
        "--models",
        "flux",
        "qwen",
        "--budgets",
        "29",
        "37",
        "41",
    ]
    import sys

    saved = sys.argv
    sys.argv = ["build_sp_cross_schedules.py", *argv]
    try:
        assert builder.main() == 0
    finally:
        sys.argv = saved

    files = sorted(path.name for path in tmp_path.glob("*.txt"))
    assert len(files) == 2 * 3 * len(builder.SCHEDULE_ORDER) == 48
    manifest = _read_tsv(tmp_path / "manifest.tsv")
    assert len(manifest) == 48

    fixed = {
        (row["model"], row["target_k"], row["method"]): row
        for row in _read_tsv(fixed_path)
    }
    native_rows = _read_tsv(counts_path)
    for row in manifest:
        key = (row["model"], row["target_k"], row["source_method"])
        if row["axis"] == "fixed":
            source = fixed[key]
        else:
            source = builder.select_native(
                native_rows,
                model=row["model"],
                target_k=int(row["target_k"]),
                method=row["source_method"],
                dataset="synthetic_discovery",
            )
            assert source is not None
        assert row["schedule"] == source["schedule"]
        assert int(row["popcount"]) == int(source["cache_count"])
        assert int(row["popcount"]) == int(row["target_k"])
        bits = (tmp_path / f"{row['model']}_k{row['target_k']}_{row['name']}.txt").read_text(
            encoding="utf-8"
        )
        assert bits.strip() == source["schedule"]
        assert bits.strip()[0] == "0"


def test_builder_uses_the_exact_k_modal_sencache_path(
    tmp_path: Path, schedule_sources: tuple[Path, Path]
) -> None:
    import sys

    fixed_path, counts_path = schedule_sources
    saved = sys.argv
    sys.argv = [
        "build_sp_cross_schedules.py",
        "--out_dir",
        str(tmp_path),
        "--fixed_paths", str(fixed_path),
        "--path_counts", str(counts_path),
        "--dataset", "synthetic_discovery",
        "--budgets",
        "29",
    ]
    try:
        assert builder.main() == 0
    finally:
        sys.argv = saved
    manifest = _read_tsv(tmp_path / "manifest.tsv")
    sencache = {
        (row["model"], row["name"]): (int(row["source_rank"]), int(row["k_vs_target"]))
        for row in manifest
        if row["name"] == "sencache_top1"
    }
    assert sencache == {
        ("flux", "sencache_top1"): (2, 0),
        ("qwen", "sencache_top1"): (2, 0),
    }


def test_select_native_returns_none_when_exact_k_subset_is_empty() -> None:
    rows = [
        {
            "model": "flux",
            "dataset": "discovery",
            "target_k": "2",
            "method": "sencache",
            "cache_count": "3",
            "count": "10",
            "schedule": "01110",
        }
    ]
    assert builder.select_native(
        rows,
        model="flux",
        target_k=2,
        method="sencache",
        dataset="discovery",
    ) is None


def test_committed_schedules_match_their_manifest() -> None:
    manifest = _read_tsv(SCHEDULE_DIR / "manifest.tsv")
    assert len(manifest) == 48
    for row in manifest:
        path = REPO_ROOT / row["path"]
        bits = path.read_text(encoding="utf-8").strip()
        assert bits == row["schedule"]
        assert bits.count("1") == int(row["popcount"]) == int(row["cache_count"])
        assert int(row["popcount"]) == int(row["target_k"])
        cache_steps = flux_load_schedule_file(path, num_steps=50)
        assert len(cache_steps) == int(row["popcount"])
        assert cache_steps == qwen_load_schedule_file(path, num_steps=50)
        assert 0 not in cache_steps
        assert cache_steps == tuple(sorted(set(cache_steps)))


@pytest.mark.parametrize(
    "loader", [flux_load_schedule_file, qwen_load_schedule_file]
)
def test_load_schedule_file_roundtrip_and_length_check(loader, tmp_path: Path) -> None:
    path = tmp_path / "sched.txt"
    path.write_text("001101\n", encoding="utf-8")
    assert loader(path, num_steps=6) == (2, 3, 5)

    path.write_text("00110\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="characters of 0/1"):
        loader(path, num_steps=6)

    path.write_text("00110x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="characters of 0/1"):
        loader(path, num_steps=6)
