from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.eval_video_metrics import (
    _parse_frame_indices,
    collect_entries,
    evaluate_entry,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"video")


def test_parse_frame_indices_all_and_explicit() -> None:
    assert _parse_frame_indices("all") is None
    assert _parse_frame_indices("0,16,32") == [0, 16, 32]
    with pytest.raises(ValueError, match="empty"):
        _parse_frame_indices("")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_frame_indices("0,-1")


def test_collect_entries_is_fail_closed_and_counts_both_sides(tmp_path: Path) -> None:
    acc = tmp_path / "acc"
    gt = tmp_path / "gt"
    acc.mkdir()
    gt.mkdir()
    _touch(acc / "video_00000.mp4")
    _touch(gt / "video_00000.mp4")
    _touch(gt / "video_00001.mp4")

    with pytest.raises(
        FileNotFoundError,
        match=r"missing_acc=2, missing_gt=1",
    ):
        collect_entries(acc, gt, ["a", "b", "c"], None, allow_missing=False)

    entries = collect_entries(acc, gt, ["a", "b", "c"], None, allow_missing=True)
    assert entries == [(0, acc / "video_00000.mp4", gt / "video_00000.mp4", "a")]


class _State:
    want: set[str] = set()


def test_evaluate_entry_rejects_empty_or_misaligned_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = (Path("acc.mp4"), Path("gt.mp4"))

    monkeypatch.setattr("evaluation.eval_video_metrics._read_selected_frames", lambda *_: [])
    with pytest.raises(ValueError, match="empty decoded video"):
        evaluate_entry(0, *paths, "prompt", None, _State())

    frames = {
        paths[0]: [np.zeros((4, 4, 3), dtype=np.uint8)],
        paths[1]: [np.zeros((5, 4, 3), dtype=np.uint8)],
    }
    monkeypatch.setattr(
        "evaluation.eval_video_metrics._read_selected_frames",
        lambda path, _indices: frames[path],
    )
    with pytest.raises(ValueError, match="frame shape mismatch"):
        evaluate_entry(0, *paths, "prompt", None, _State())
