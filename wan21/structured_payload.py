"""Structured Wan2.1 payload helpers for fine and segment cache experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import torch

from lib.hermite import hermite_update, hicache_predict
from lib.taylor import taylor_predict


FINE_PAYLOAD_MODES = (
    "fine_reuse",
    "fine_taylor_o1",
    "fine_taylor_o2",
    "fine_hicache_o2",
)

SEGMENT_PAYLOAD_MODES = (
    "segment_reuse",
    "segment_taylor_o1",
    "segment_taylor_o2",
    "segment_hicache_o2",
    "segment_ensemble_mean",
)

SEGMENT_LAYOUTS = (
    "seg2",
    "seg4",
    "seg8",
    "seg16",
    "block32",
)

History = Dict[int, torch.Tensor]
SegmentSpec = Dict[str, int]


def payload_base_mode(mode: str) -> str:
    return {
        "fine_reuse": "reuse",
        "fine_taylor_o1": "taylor_o1",
        "fine_taylor_o2": "taylor_o2",
        "fine_hicache_o2": "hicache_o2",
        "segment_reuse": "reuse",
        "segment_taylor_o1": "taylor_o1",
        "segment_taylor_o2": "taylor_o2",
        "segment_hicache_o2": "hicache_o2",
        "segment_ensemble_mean": "ensemble_mean",
    }[str(mode)]


def payload_max_order(mode: str) -> int:
    base = payload_base_mode(mode)
    if base == "reuse":
        return 0
    if base == "taylor_o1":
        return 1
    return 2


def available_order(history: Optional[History]) -> int:
    if not history:
        return -1
    return max(int(k) for k in history.keys())


def update_history(
    prev_history: Optional[History],
    feature: torch.Tensor,
    *,
    step_gap: int,
    max_order: int,
) -> History:
    return {
        int(k): v.detach().clone()
        for k, v in hermite_update(
            prev_history,
            feature.detach(),
            step_gap=max(1, int(step_gap)),
            max_order=int(max_order),
        ).items()
    }


def predict_from_history(
    history: History,
    *,
    step_offset: int,
    mode: str,
    sigma: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not history or 0 not in history:
        raise RuntimeError("payload history is missing order-0 feature")

    base = payload_base_mode(mode)
    requested_order = payload_max_order(mode)
    order_avail = available_order(history)
    effective_order = min(int(requested_order), int(order_avail))

    if base == "reuse":
        out = history[0]
    elif base == "taylor_o1":
        out = taylor_predict(history, int(step_offset), max_order=1)
    elif base == "taylor_o2":
        out = taylor_predict(history, int(step_offset), max_order=2)
    elif base == "hicache_o2":
        out = hicache_predict(history, int(step_offset), sigma=float(sigma), max_order=2)
    elif base == "ensemble_mean":
        parts = [
            taylor_predict(history, int(step_offset), max_order=1),
            taylor_predict(history, int(step_offset), max_order=2),
            hicache_predict(history, int(step_offset), sigma=float(sigma), max_order=2),
        ]
        out = torch.stack(parts, dim=0).mean(dim=0)
    else:
        raise ValueError(f"unsupported payload base mode: {base!r}")

    return out, {
        "payload_base_mode": base,
        "payload_requested_order": int(requested_order),
        "payload_available_order": int(order_avail),
        "payload_effective_order": int(effective_order),
        "payload_order_degraded": bool(effective_order < requested_order),
    }


def split_ranges(n: int, parts: int) -> list[tuple[int, int]]:
    if n <= 0 or parts <= 0:
        return []
    out: list[tuple[int, int]] = []
    for i in range(parts):
        start = int(round(i * n / parts))
        end = int(round((i + 1) * n / parts))
        if start < end:
            out.append((start, end))
    return out


def build_segments(layout: str, *, num_blocks: int) -> list[SegmentSpec]:
    layout = str(layout)
    if layout not in SEGMENT_LAYOUTS:
        raise ValueError(f"unknown Wan2.1 segment layout: {layout!r}")
    if layout == "block32":
        ranges = [(i, i + 1) for i in range(int(num_blocks))]
    else:
        parts = int(layout.removeprefix("seg"))
        ranges = split_ranges(int(num_blocks), parts)
    return [{"start": int(start), "end": int(end)} for start, end in ranges]


def segment_key(spec: SegmentSpec) -> tuple[int, int]:
    return int(spec["start"]), int(spec["end"])


def segment_id(spec: SegmentSpec) -> str:
    start, end = segment_key(spec)
    return f"blocks:{start}-{end}"


def segment_spec_hash(specs: list[SegmentSpec]) -> str:
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def norm_optional(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().to(torch.float32).norm().item())
