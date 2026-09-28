#!/usr/bin/env python3
"""Fit and optionally validate TeaCache's degree-four scalar rescaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    return parser.parse_args()


def _rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def _finite_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["input_rel_l1"] for row in rows], dtype=np.float64)
    y = np.asarray([row["output_rel_l1"] for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _metrics(rows: list[dict], coefficients: np.ndarray) -> dict[str, float | int]:
    x, y = _finite_arrays(rows)
    if y.size == 0:
        raise ValueError("TeaCache metric set has no finite input/output pairs")
    prediction = np.polyval(coefficients, x)
    residual = prediction - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    train = _rows(args.train)
    x, y = _finite_arrays(train)
    if x.size <= int(args.degree):
        raise ValueError(
            f"TeaCache fit needs more than {args.degree} finite pairs, got {x.size}"
        )
    coefficients = np.polyfit(x, y, deg=args.degree)
    payload = {
        "format": "teacache_polynomial.v1",
        "model": "qwen_image",
        "degree": args.degree,
        "coefficients": [float(value) for value in coefficients],
        "train": _metrics(train, coefficients),
        "holdout": (
            None if not args.holdout else _metrics(_rows(args.holdout), coefficients)
        ),
        "train_files": [str(path) for path in args.train],
        "holdout_files": [str(path) for path in args.holdout],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
