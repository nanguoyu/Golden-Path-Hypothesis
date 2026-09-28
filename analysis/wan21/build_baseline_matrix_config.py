#!/usr/bin/env python3
"""Freeze the Wan2.1 baseline-matrix configuration (plan section 3, gate P5).

Collects the threshold sweeps of P3 and the schedule searches of P4 into
`resources/wan21/baseline_matrix_config.v1.json`, validates that every one of
the 9 methods x 2 datasets x 3 cache tiers resolves, self-hashes the result and
writes it immutably. Nothing is searched or defaulted here: every number arrives
on the command line or inside a schedule file produced by the search that owns
it.

    python analysis/wan21/build_baseline_matrix_config.py \\
        --threshold seacache:penguin599:K29=0.0731 \\
        --threshold seacache:vbench944:K29=0.0698 ... \\
        --method-param seacache:penguin599:K29:first_enhance=1 \\
        --method-param seacache:penguin599:K29:power_exp=3.0 ... \\
        --asset sencache:sencache_sensitivity_path=$DATA/wan21/sencache/table.npz \\
        --asset l2p:l2p_weights=$DATA/wan21/l2p/weights.pt \\
        --schedule budcache:K29=$DATA/wan21/budcache/k29.json ... \\
        --schedule meancache:K29=$DATA/wan21/meancache/k29.json ... \\
        --shared-schedule K29=$DATA/wan21/shared/k29.json ...

Every knob a method may freeze (`wan21.matrix_config.METHOD_PARAMS`) is required
for every one of its cells, so the sweep that picks a threshold also states the
values it swept at. That is what makes the frozen file readable on its own:
without it a cell silently runs whatever the runner's argparse default happens
to be, which the file does not record and no result can be traced to.

`--asset` freezes the digest, not the path: the fitted files live under $DATA and
the matrix runs on more than one cluster. The runner hashes whatever path it is
given and refuses a mismatch.

The four in-repo prompt files are frozen the same way but are not flags, which
is the one place this diverges from the HunyuanVideo twin: `wan21/matrix_config`
already declares them (`EVALUATION_MANIFESTS`, `CALIBRATION_PROMPTS`), because
plan sections 1.2 and 3.1 reuse the Hunyuan lane's frozen files in place rather
than copying them into `resources/wan21/`. Taking them on the command line as
well would let the digests in the frozen file disagree with the constants the
runner and the sweep read, which is the failure the digests exist to catch.

Re-running with identical inputs is a no-op; re-running with different ones
fails, because the frozen configuration must not change under a matrix that is
already running (`write_immutable_json`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Unconditional, never guarded: `analysis/wan21/` shadows the repo-root `wan21/`
# package and a wrapper's PYTHONPATH puts the root behind the shadow, which is
# what made the guarded form a silent no-op in commit b35a0d3.
sys.path.insert(0, str(ROOT))

from hunyuan_video.config import hash_json  # noqa: E402
from hunyuan_video.records import sha256_file, write_immutable_json  # noqa: E402
from wan21.matrix_config import (  # noqa: E402
    BUDGETS,
    CALIBRATION_PROMPTS,
    CONFIG_PATH,
    DATASETS,
    DYNAMIC_METHODS,
    EVALUATION_MANIFESTS,
    HASH_FIELD,
    METHOD_ASSETS,
    METHOD_PARAMS,
    METHODS,
    NUM_STEPS,
    PROTOCOL_ID,
    SCHEMA,
    load_matrix_config,
    validate_payload,
)
from wan21.methods_glue import SHARED_SCHEDULE_METHODS  # noqa: E402

# Methods whose schedule is searched per tier and belongs to them alone; the
# other three fixed-schedule methods share one table (plan section 3.3).
SCHEDULE_METHOD_ARGS = ("budcache", "meancache")

# The two dataset-keyed file bindings the frozen payload carries, and the payload
# key each lands under. Both are read in place from `wan21/matrix_config`.
FILE_BINDINGS = {
    "threshold_calibration": CALIBRATION_PROMPTS,
    "evaluation_datasets": EVALUATION_MANIFESTS,
}


def _threshold(value: str) -> tuple[str, str, str, float]:
    key, _, number = value.partition("=")
    parts = key.split(":")
    if (not number or len(parts) != 3 or parts[0] not in DYNAMIC_METHODS
            or parts[1] not in DATASETS or parts[2] not in BUDGETS):
        raise argparse.ArgumentTypeError(
            f"--threshold must look like "
            f"{DYNAMIC_METHODS[0]}:{DATASETS[0]}:{BUDGETS[0]}=0.12, got {value!r}"
        )
    return parts[0], parts[1], parts[2], float(number)


def _method_param(value: str) -> tuple[str, str, str, str, float]:
    key, _, number = value.partition("=")
    parts = key.split(":")
    if (not number or len(parts) != 4 or parts[0] not in METHOD_PARAMS
            or parts[1] not in DATASETS or parts[2] not in BUDGETS
            or parts[3] not in METHOD_PARAMS[parts[0]]):
        freezable = {method: sorted(spec) for method, spec in METHOD_PARAMS.items() if spec}
        raise argparse.ArgumentTypeError(
            f"--method-param must look like method:dataset:budget:knob=value with the "
            f"knob one its method freezes ({freezable}), got {value!r}")
    return parts[0], parts[1], parts[2], parts[3], float(number)


def _asset(value: str) -> tuple[str, str, Path]:
    key, _, path = value.partition("=")
    method, _, name = key.partition(":")
    if not path or method not in METHOD_ASSETS or name not in METHOD_ASSETS[method]:
        known = {method: list(names) for method, names in METHOD_ASSETS.items()}
        raise argparse.ArgumentTypeError(
            f"--asset must look like method:argument=path, naming a file input its "
            f"method reads ({known}), got {value!r}")
    return method, name, Path(path)


def _schedule(value: str) -> tuple[str, str, Path]:
    key, _, path = value.partition("=")
    method, _, budget = key.partition(":")
    if not path or method not in SCHEDULE_METHOD_ARGS or budget not in BUDGETS:
        raise argparse.ArgumentTypeError(
            f"--schedule must look like meancache:{BUDGETS[0]}=path.json, got {value!r}"
        )
    return method, budget, Path(path)


def _shared_schedule(value: str) -> tuple[str, Path]:
    budget, _, path = value.partition("=")
    if not path or budget not in BUDGETS:
        raise argparse.ArgumentTypeError(
            f"--shared-schedule must look like {BUDGETS[0]}=path.json, got {value!r}"
        )
    return budget, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=_threshold, action="append", default=[])
    parser.add_argument("--method-param", type=_method_param, action="append", default=[])
    parser.add_argument("--asset", type=_asset, action="append", default=[])
    parser.add_argument("--schedule", type=_schedule, action="append", default=[])
    parser.add_argument("--shared-schedule", type=_shared_schedule, action="append",
                        default=[])
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--output", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def _read_table(path: Path, *, with_spans: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cache_steps" not in payload:
        raise ValueError(f"schedule file has no cache_steps: {path}")
    table: dict[str, Any] = {
        "cache_steps": [int(step) for step in payload["cache_steps"]],
        "source_file": str(path),
        "source_sha256": sha256_file(path),
    }
    if with_spans:
        spans = payload.get("jvp_spans")
        if not isinstance(spans, dict) or not spans:
            raise ValueError(f"MeanCache schedule file carries no jvp_spans: {path}")
        table["jvp_spans"] = {str(int(step)): int(span) for step, span in spans.items()}
    return table


def _file_binding(name: str, files: dict[str, Path]) -> dict[str, dict[str, str]]:
    """`dataset -> {file, sha256}` for one of the two in-repo prompt bindings."""

    rows: dict[str, dict[str, str]] = {}
    for dataset, path in files.items():
        if dataset not in DATASETS:
            raise ValueError(f"{name}: {dataset} is not a matrix dataset")
        if not path.is_file():
            raise ValueError(f"{name}: {dataset} file does not exist: {path}")
        rows[dataset] = {
            "file": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    if set(rows) != set(DATASETS):
        raise ValueError(f"{name} must cover exactly {list(DATASETS)}")
    return rows


def build_payload(
    *,
    thresholds: list[tuple[str, str, str, float]],
    method_params: list[tuple[str, str, str, str, float]],
    assets: list[tuple[str, str, Path]],
    schedules: list[tuple[str, str, Path]],
    shared_schedules: list[tuple[str, Path]],
    protocol_id: str,
) -> dict[str, Any]:
    threshold_map: dict[str, dict[str, dict[str, float]]] = {}
    for method, dataset, budget, value in thresholds:
        per_dataset = threshold_map.setdefault(method, {}).setdefault(dataset, {})
        if budget in per_dataset:
            raise ValueError(f"threshold given twice for ({method}, {dataset}, {budget})")
        per_dataset[budget] = float(value)

    param_map: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for method, dataset, budget, name, value in method_params:
        knobs = (param_map.setdefault(method, {}).setdefault(dataset, {})
                 .setdefault(budget, {}))
        if name in knobs:
            raise ValueError(f"{name} given twice for ({method}, {dataset}, {budget})")
        knobs[name] = float(value)

    asset_map: dict[str, dict[str, str]] = {}
    for method, name, path in assets:
        per_method = asset_map.setdefault(method, {})
        if name in per_method:
            raise ValueError(f"{name} given twice for {method}")
        if not path.is_file():
            raise ValueError(f"{method} {name} is not a file: {path}")
        per_method[name] = sha256_file(path)

    tables: dict[str, dict[str, Any]] = {}
    schedule_map: dict[str, dict[str, str]] = {}
    for method, budget, path in schedules:
        table_id = f"{method}_{budget}"
        if table_id in tables:
            raise ValueError(f"schedule given twice for ({method}, {budget})")
        tables[table_id] = _read_table(path, with_spans=(method == "meancache"))
        schedule_map.setdefault(method, {})[budget] = table_id
    for budget, path in shared_schedules:
        table_id = f"shared_{budget}"
        if table_id in tables:
            raise ValueError(f"shared schedule given twice for {budget}")
        # A MeanCache search output satisfies the shared table's forbidden steps
        # too, and `_read_table` would quietly drop the `jvp_spans` that are half
        # of its solved path -- three methods would then be running MeanCache's
        # schedule under their own names.
        if "jvp_spans" in json.loads(path.read_text(encoding="utf-8")):
            raise ValueError(
                f"shared schedule {path} carries jvp_spans, so it is a MeanCache "
                f"solution; the shared table is not one")
        tables[table_id] = _read_table(path, with_spans=False)
        for method in SHARED_SCHEDULE_METHODS:
            schedule_map.setdefault(method, {})[budget] = table_id

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "num_steps": NUM_STEPS,
        "budgets": {budget: int(budget[1:]) for budget in BUDGETS},
        "thresholds": threshold_map,
        "method_params": param_map,
        "method_assets": asset_map,
        "schedule_tables": tables,
        "schedules": schedule_map,
    }
    for key, files in FILE_BINDINGS.items():
        payload[key] = _file_binding(key, files)
    validate_payload(payload)
    payload[HASH_FIELD] = hash_json(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        thresholds=args.threshold,
        method_params=args.method_param,
        assets=args.asset,
        schedules=args.schedule,
        shared_schedules=args.shared_schedule,
        protocol_id=args.protocol_id,
    )
    write_immutable_json(args.output, payload)
    config = load_matrix_config(args.output)
    cells = {}
    for method in METHODS:
        for dataset in DATASETS:
            for budget in BUDGETS:
                entry = config.entry(method, budget, dataset)
                cells[f"{method}:{dataset}:{budget}"] = (
                    entry.threshold if entry.threshold is not None else entry.table_id
                )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_id": payload["protocol_id"],
                HASH_FIELD: config.sha256,
                "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "thresholds": sum(len(per_budget)
                                  for per_dataset in payload["thresholds"].values()
                                  for per_budget in per_dataset.values()),
                "method_params": sum(len(knobs)
                                     for per_dataset in payload["method_params"].values()
                                     for per_budget in per_dataset.values()
                                     for knobs in per_budget.values()),
                "method_assets": sum(len(row) for row in payload["method_assets"].values()),
                "file_bindings": {
                    key: {dataset: row["sha256"] for dataset, row in payload[key].items()}
                    for key in FILE_BINDINGS
                },
                "schedule_tables": len(payload["schedule_tables"]),
                "schedule_cells": sum(len(row) for row in payload["schedules"].values()),
                "cells": dict(sorted(cells.items())),
            },
            indent=2,
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
