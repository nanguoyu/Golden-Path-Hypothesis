from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flux.exhaustive_k41_runner import (
    PART_SCHEMA,
    _file_sha256,
    _prompt_text_sha256,
    mse_and_psnr,
    validate_existing_part,
)


def test_mse_and_psnr_uses_uint8_data_range_255() -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = np.ones((2, 2, 3), dtype=np.uint8) * 10
    mse, psnr = mse_and_psnr(reference, candidate)
    assert mse == 100.0
    assert psnr == pytest.approx(28.1308036087)


def test_mse_and_psnr_exact_match_is_finite() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert mse_and_psnr(image, image) == (0.0, 120.0)


def test_resume_validation_rejects_short_part(tmp_path: Path) -> None:
    path = tmp_path / "part.json"
    path.write_text(
        json.dumps(
            {
                "schema": PART_SCHEMA,
                "experiment_fingerprint": "abc",
                "rank_start": 10,
                "rank_end": 12,
                "rows": [{"rank": 10}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_existing_part(
            path, rank_start=10, rank_end=12, experiment_fingerprint="abc"
        )


def test_conditioning_identity_hashes_unicode_and_file_bytes(tmp_path: Path) -> None:
    prompts = ("an avocado", "a store front with ‘openai’ written on it")
    assert _prompt_text_sha256(prompts) == _prompt_text_sha256(prompts)
    assert _prompt_text_sha256(prompts) != _prompt_text_sha256(tuple(reversed(prompts)))
    artifact = tmp_path / "conditioning.pt"
    artifact.write_bytes(b"canonical-conditioning")
    assert (
        _file_sha256(artifact)
        == "d13110aedb7b01050a7e96449ac3b0e384d0bfc3acb55ec1ca426e83fa219653"
    )
