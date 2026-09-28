#!/usr/bin/env python3
"""Build an exact-NFE MeanCache path from calibration edge costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost_shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int, default=29)
    parser.add_argument(
        "--model",
        choices=["flux", "qwen_image", "hunyuan_video", "wan21"],
        default="flux",
    )
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--first_full_steps", type=int, default=5)
    parser.add_argument("--last_full_steps", type=int, default=1)
    return parser.parse_args()


def _load(
    paths: list[Path],
    *,
    model: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int, int | None]:
    sums = None
    counts = None
    spans = None
    num_steps = None
    max_edge_gap = None
    for path in paths:
        shard = np.load(path)
        # Cost arrays from different backbones are shape-identical, so nothing but
        # this tag can tell them apart. The HunyuanVideo and Wan2.1 collectors
        # always write it; untagged shards can only come from the older FLUX /
        # Qwen collectors.
        tag = str(shard["model"]) if "model" in shard else None
        if tag is not None and tag != model:
            raise ValueError(f"{path} was collected on {tag}, not {model}")
        if tag is None and model in ("hunyuan_video", "wan21"):
            raise ValueError(f"{path} carries no backbone tag, so it is not a {model} shard")
        current_spans = tuple(int(value) for value in shard["jvp_spans"])
        current_steps = int(shard["num_steps"])
        # Every collector records the gap it was run with; shards written before
        # the field existed still load, they just carry it through as unknown.
        current_gap = int(shard["max_edge_gap"]) if "max_edge_gap" in shard else None
        if spans is None:
            spans = current_spans
            num_steps = current_steps
            max_edge_gap = current_gap
            sums = np.asarray(shard["cost_sums"], dtype=np.float64)
            counts = np.asarray(shard["cost_counts"], dtype=np.int64)
        else:
            if current_spans != spans:
                raise ValueError("MeanCache cost shards use different JVP spans")
            if current_steps != num_steps or current_gap != max_edge_gap:
                raise ValueError("MeanCache cost shards use different trajectory geometry")
            sums += shard["cost_sums"]
            counts += shard["cost_counts"]
    assert sums is not None and counts is not None and spans is not None
    assert num_steps is not None
    return sums, counts, spans, num_steps, max_edge_gap


def _select(
    mean_cost: np.ndarray,
    spans: tuple[int, ...],
    *,
    num_steps: int,
    full_count: int,
    gamma: float,
    first_full_steps: int,
    last_full_steps: int,
) -> tuple[tuple[int, ...], list[tuple[int, int, int, float]]]:
    mandatory = set(range(first_full_steps))
    mandatory.update(range(num_steps - last_full_steps, num_steps))
    mandatory.add(0)
    nodes = num_steps + 1
    sentinel = num_steps
    edge_count = full_count
    dp = np.full((edge_count + 1, nodes), np.inf)
    parent = np.full((edge_count + 1, nodes), -1, dtype=np.int64)
    span_choice = np.full((edge_count + 1, nodes), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for used in range(1, edge_count + 1):
        for destination in range(used, nodes):
            for source in range(used - 1, destination):
                if not np.isfinite(dp[used - 1, source]):
                    continue
                if any(step in mandatory for step in range(source + 1, destination)):
                    continue
                costs = mean_cost[:, source, destination]
                choice = int(np.argmin(costs))
                edge = costs[choice]
                if not np.isfinite(edge):
                    continue
                candidate = dp[used - 1, source] + float(edge) ** float(gamma)
                if candidate < dp[used, destination]:
                    dp[used, destination] = candidate
                    parent[used, destination] = source
                    span_choice[used, destination] = choice

    if not np.isfinite(dp[edge_count, sentinel]):
        raise RuntimeError("MeanCache path search could not reach the terminal node")
    edges: list[tuple[int, int, int, float]] = []
    destination = sentinel
    full_steps = []
    for used in range(edge_count, 0, -1):
        source = int(parent[used, destination])
        choice = int(span_choice[used, destination])
        if source < 0 or choice < 0:
            raise RuntimeError("MeanCache path backtracking failed")
        edges.append(
            (
                source,
                destination,
                spans[choice],
                float(mean_cost[choice, source, destination]),
            )
        )
        full_steps.append(source)
        destination = source
    full = tuple(sorted(full_steps))
    if len(full) != full_count or not mandatory.issubset(full):
        raise RuntimeError("MeanCache path violates the exact full-step budget")
    return full, list(reversed(edges))


def main() -> int:
    args = parse_args()
    sums, counts, spans, shard_steps, max_edge_gap = _load(args.cost_shards, model=args.model)
    if shard_steps != args.num_steps:
        raise SystemExit(
            f"cost shards were collected over {shard_steps} steps, not {args.num_steps}"
        )
    mean = np.full_like(sums, np.inf, dtype=np.float64)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    full_count = args.num_steps - args.cache_count
    full, edges = _select(
        mean,
        spans,
        num_steps=args.num_steps,
        full_count=full_count,
        gamma=args.gamma,
        first_full_steps=args.first_full_steps,
        last_full_steps=args.last_full_steps,
    )
    full_set = set(full)
    cache_steps = [
        step for step in range(args.num_steps) if step not in full_set
    ]
    jvp_spans: dict[str, int] = {}
    for source, destination, span, _cost in edges:
        for step in range(source + 1, destination):
            if step in cache_steps:
                jvp_spans[str(step)] = span
    payload = {
        "format": f"{args.model.replace('_', '-')}-meancache-schedule-v1",
        "model": args.model,
        "num_steps": args.num_steps,
        "cache_count": args.cache_count,
        "full_count": full_count,
        "cache_steps": cache_steps,
        "full_steps": list(full),
        "jvp_spans": jvp_spans,
        "gamma": args.gamma,
        "first_full_steps": args.first_full_steps,
        "last_full_steps": args.last_full_steps,
        "max_edge_gap": max_edge_gap,
        "cost_shards": [str(path) for path in args.cost_shards],
        "edges": [
            {
                "source": source,
                "destination": destination,
                "jvp_span": span,
                "mean_l1_cost": cost,
            }
            for source, destination, span, cost in edges
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
