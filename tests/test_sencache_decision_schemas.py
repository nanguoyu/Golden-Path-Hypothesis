"""One reader for the three decision schemas the lanes write.

The bug this guards: `collect` read FLUX's `per_step` key only, so the first run
over Qwen's files died with KeyError('per_step') after the sweep had already
spent the GPU time. FLUX writes `per_step`/`u`, Qwen writes `steps`/`u`, Wan
writes `records`/`action`; all three describe the same 50 decisions.

Plan: docs/sencache_recalibration_plan_zh.md S1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.sencache_frontier import _decision_bits  # noqa: E402

#: the same trajectory: full for ten steps, then alternating, terminal full
BITS = "0000000000" + "10" * 19 + "01"
assert len(BITS) == 50


def _flux(bits: str) -> dict:
    return {"schema": "flux_native_gate_decisions.v1",
            "per_step": [{"step": i, "u": int(b), "action": "cache" if b == "1" else "full"}
                         for i, b in enumerate(bits)]}


def _qwen(bits: str) -> dict:
    return {"schema": "qwen_image_decisions.v1",
            "steps": [{"step": i, "u": int(b), "action": "cache" if b == "1" else "full"}
                      for i, b in enumerate(bits)]}


def _wan(bits: str) -> dict:
    return {"records": [{"step": i, "action": "cache" if b == "1" else "full"}
                        for i, b in enumerate(bits)]}


@pytest.mark.parametrize("build", [_flux, _qwen, _wan], ids=["flux", "qwen", "wan21"])
def test_every_lane_reads_to_the_same_bits(build) -> None:
    assert _decision_bits(build(BITS)) == BITS


def test_rows_out_of_order_are_sorted_by_step() -> None:
    payload = _flux(BITS)
    payload["per_step"] = list(reversed(payload["per_step"]))
    assert _decision_bits(payload) == BITS


def test_a_payload_with_no_known_row_key_fails_loudly() -> None:
    with pytest.raises(KeyError, match="carries none of"):
        _decision_bits({"schema": "something_new.v1", "rows": [{"step": 0, "u": 1}]})


def test_a_row_with_no_known_decision_key_fails_loudly() -> None:
    with pytest.raises(KeyError, match="neither 'u' nor 'action'"):
        _decision_bits({"per_step": [{"step": 0, "cached": True}]})


def test_an_empty_row_list_is_not_silently_an_empty_run() -> None:
    with pytest.raises(KeyError):
        _decision_bits({"per_step": []})
